# src_cnn_v2/poc_two_stream_real_time_detection.py
"""
Proof of Concept (POC) for a Two-Stream Real-Time Video Stabilization and Leak Detection System.
(Version 9: Added dynamic crop visualization and CPU-fallback for Huber regression)
"""
import os
import sys
import argparse
import time
import threading
import numpy as np
import scipy.io
import cv2
import imageio.v2 as imageio
import math
from collections import deque
from scipy import stats
from skimage.feature import peak_local_max
from tqdm import tqdm
from joblib import Parallel, delayed

try:
    import torch
    from sklearn.linear_model import HuberRegressor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- Core Helper Functions ---


def add_gradual_shake(frames, args):
    print("--- Generating simulated shaky video with custom parameters... ---")
    H, W, T = frames.shape
    shaky_frames = np.zeros_like(frames)
    start_shift = (0.0, 0.0)
    shift1_target = (int(W * (args.shift1_x / 100.0)),
                     int(H * (args.shift1_y / 100.0)))
    shift2_target = (shift1_target[0] + int(W * (args.shift2_x / 100.0)),
                     shift1_target[1] + int(H * (args.shift2_y / 100.0)))
    total_x_shifts, total_y_shifts = [], []
    phase1_len = args.shift1_frame
    if phase1_len > 0:
        total_x_shifts.extend(
            list(np.linspace(start_shift[0], shift1_target[0], num=phase1_len)))
        total_y_shifts.extend(
            list(np.linspace(start_shift[1], shift1_target[1], num=phase1_len)))
    phase2_len = args.shift2_frame - args.shift1_frame
    if phase2_len > 0:
        total_x_shifts.extend(
            list(np.linspace(total_x_shifts[-1], shift2_target[0], num=phase2_len + 1))[1:])
        total_y_shifts.extend(
            list(np.linspace(total_y_shifts[-1], shift2_target[1], num=phase2_len + 1))[1:])
    phase3_len = T - args.shift2_frame
    if phase3_len > 0:
        total_x_shifts.extend([total_x_shifts[-1]] * phase3_len)
        total_y_shifts.extend([total_y_shifts[-1]] * phase3_len)
    for i in range(T):
        M = np.float32([[1, 0, total_x_shifts[i]], [0, 1, total_y_shifts[i]]])
        shaky_frames[:, :, i] = cv2.warpAffine(frames[:, :, i], M, (W, H))
    return shaky_frames

# CPU theil-sen function


def _calculate_theilsen_slope_for_row(row_data, t):
    W = row_data.shape[0]
    row_slopes = np.zeros(W, dtype=np.float32)
    for c in range(W):
        try:
            res = stats.theilslopes(row_data[c, :], t, 0.9)
            row_slopes[c] = res[0] if np.isfinite(res[0]) else 0.0
        except (ValueError, IndexError):
            row_slopes[c] = 0.0
    return row_slopes

# CPU Huber function


def _calculate_huber_slope_for_row_cpu(row_data, t):
    W = row_data.shape[0]
    row_slopes = np.zeros(W, dtype=np.float32)
    t_reshaped = t.reshape(-1, 1)
    huber = HuberRegressor()
    for c in range(W):
        try:
            huber.fit(t_reshaped, row_data[c, :])
            row_slopes[c] = huber.coef_[
                0] if np.isfinite(huber.coef_[0]) else 0.0
        except ValueError:
            row_slopes[c] = 0.0
    return row_slopes


def calculate_temporal_trend_map_cpu(frames, regressor):
    H, W, T = frames.shape
    t = np.arange(T)
    print(f"\n--- (Slow Path) Using CPU Parallel {regressor.upper()} ---")

    func_to_parallelize = _calculate_theilsen_slope_for_row
    if regressor == 'huber':
        func_to_parallelize = _calculate_huber_slope_for_row_cpu

    results = Parallel(n_jobs=-1)(
        delayed(func_to_parallelize)(frames[r, :, :], t)
        for r in tqdm(range(H), desc=f"  - CPU {regressor.upper()} Progress", file=sys.stdout))
    slope_map = np.vstack(results)
    np.nan_to_num(slope_map, copy=False, nan=0.0)
    return np.maximum(slope_map, 0.0)


def calculate_local_heat_z_score(frames):
    temp_mean = np.mean(frames, axis=2).astype(np.float32)
    loc_mu = cv2.GaussianBlur(temp_mean, (31, 31), 0)
    temp_mean_sq = cv2.GaussianBlur(temp_mean**2, (31, 31), 0)
    loc_sd = np.sqrt(np.maximum(temp_mean_sq - loc_mu**2, 0)) + 1e-9
    return np.maximum((temp_mean - loc_mu) / loc_sd, 0.0)


def find_top_n_leaks(score_map, num_leaks):
    min_score_threshold = score_map.max() * 0.10
    coordinates = peak_local_max(
        score_map, min_distance=50, threshold_abs=min_score_threshold)
    if coordinates.size == 0:
        return []
    scores = score_map[coordinates[:, 0], coordinates[:, 1]]
    candidates = [{'coords': (r, c), 'score': s}
                  for (r, c), s in zip(coordinates, scores)]
    candidates.sort(key=lambda p: p['score'], reverse=True)
    return candidates[:num_leaks]


class OnlineOLSCalculator:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.n = 0
        self.sum_x = np.float64(0)
        self.sum_y = np.zeros((H, W), dtype=np.float64)
        self.sum_xy = np.zeros((H, W), dtype=np.float64)
        self.sum_x2 = np.float64(0)

    def add_frame(self, frame, mask):
        t = self.n
        self.sum_x += t
        self.sum_y[mask] += frame[mask]
        self.sum_xy[mask] += frame[mask] * t
        self.sum_x2 += t**2
        self.n += 1

    def calculate_slope(self):
        if self.n < 2:
            return np.zeros((self.H, self.W), dtype=np.float32)
        numerator = self.n * self.sum_xy - self.sum_x * self.sum_y
        denominator = self.n * self.sum_x2 - self.sum_x**2
        if denominator == 0:
            return np.zeros((self.H, self.W), dtype=np.float32)
        slope = np.zeros_like(self.sum_y, dtype=np.float32)
        valid_pixels = self.sum_y != 0
        slope[valid_pixels] = numerator[valid_pixels] / denominator
        np.nan_to_num(slope, copy=False, nan=0.0)
        return np.maximum(slope, 0.0).astype(np.float32)


class SlowPathAnalyzer(threading.Thread):
    def __init__(self, num_leaks, analysis_interval_frames, regressor, use_gpu, compare_mode):
        super().__init__()
        self.daemon = True
        self.frame_buffer = deque()
        self.num_leaks = num_leaks
        self.analysis_interval = analysis_interval_frames
        self.regressor = regressor
        self.use_gpu = use_gpu
        self.compare_mode = compare_mode
        self.results = {}
        self._stop_event = threading.Event()

    def add_frame(self, frame): self.frame_buffer.append(frame)

    def calculate_slope_map_huber_gpu(self, frames_np):
        print(f"\n--- (Slow Path) Using GPU Accelerated Huber ---")
        device = torch.device("cuda")
        H, W, T = frames_np.shape
        frames_gpu = torch.from_numpy(frames_np).to(
            device, dtype=torch.float32)
        t_gpu = torch.arange(
            T, device=device, dtype=torch.float32).view(1, 1, -1)
        t_mean = torch.mean(t_gpu)
        frames_mean = torch.mean(frames_gpu, dim=2, keepdim=True)
        slope = torch.sum((t_gpu - t_mean) * (frames_gpu - frames_mean),
                          dim=2) / torch.sum((t_gpu - t_mean)**2, dim=2)
        intercept = frames_mean.squeeze(2) - slope * t_mean
        for _ in range(5):
            predicted = slope.unsqueeze(2) * t_gpu + intercept.unsqueeze(2)
            residuals = frames_gpu - predicted
            mad = torch.median(torch.abs(residuals - torch.median(residuals)))
            delta = 1.345 * mad
            weights = torch.ones_like(residuals)
            outliers = torch.abs(residuals) > delta
            weights[outliers] = delta / torch.abs(residuals[outliers])
            w_sum = torch.sum(weights, dim=2)
            wx_sum = torch.sum(weights * t_gpu, dim=2)
            wy_sum = torch.sum(weights * frames_gpu, dim=2)
            wxy_sum = torch.sum(weights * t_gpu * frames_gpu, dim=2)
            wx2_sum = torch.sum(weights * t_gpu**2, dim=2)
            w_denom = w_sum * wx2_sum - wx_sum**2
            slope = (w_sum * wxy_sum - wx_sum * wy_sum) / w_denom
            intercept = (wy_sum * wx2_sum - wx_sum * wxy_sum) / w_denom
        return torch.clamp(slope, min=0).cpu().numpy()

    def _run_analysis(self, frames_np, regressor_type):
        start_time = time.perf_counter()
        device_used = "CPU (Parallel)"

        if regressor_type == 'huber':
            if self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                temporal_map = self.calculate_slope_map_huber_gpu(frames_np)
                device_used = f"GPU ({torch.cuda.get_device_name(0)})"
            else:
                temporal_map = calculate_temporal_trend_map_cpu(
                    frames_np, 'huber')
        else:
            temporal_map = calculate_temporal_trend_map_cpu(
                frames_np, 'theilsen')

        heat_map = calculate_local_heat_z_score(frames_np)
        score_map = temporal_map * (heat_map ** 1.4)
        leaks_found = find_top_n_leaks(score_map, self.num_leaks)
        end_time = time.perf_counter()

        return {"time": end_time - start_time, "leaks": leaks_found, "score_map": score_map, "device": device_used}

    def run(self):
        print(
            f"--- (Slow Path) Background thread started, waiting for {self.analysis_interval} frames... ---")
        while len(self.frame_buffer) < self.analysis_interval:
            if self._stop_event.is_set():
                return
            time.sleep(0.1)

        frames_to_process = list(self.frame_buffer)
        all_frames_np = np.stack([f['frame']
                                 for f in frames_to_process], axis=2)
        all_transforms = [f['transform'] for f in frames_to_process]
        dx_vals = [t[0] for t in all_transforms]
        dy_vals = [t[1] for t in all_transforms]
        y_start = math.ceil(max(dy_vals))
        y_end = all_frames_np.shape[0] + math.floor(min(dy_vals))
        x_start = math.ceil(max(dx_vals))
        x_end = all_frames_np.shape[1] + math.floor(min(dx_vals))
        cropped_frames_np = all_frames_np[y_start:y_end, x_start:x_end, :]
        self.crop_dims = cropped_frames_np.shape[:2]

        if self.compare_mode:
            self.results['theilsen'] = self._run_analysis(
                cropped_frames_np, 'theilsen')
            self.results['huber'] = self._run_analysis(
                cropped_frames_np, 'huber')
        else:
            self.results[self.regressor] = self._run_analysis(
                cropped_frames_np, self.regressor)

    def stop(self): self._stop_event.set()


def create_comparison_gif(frames1, label1, frames2, label2, save_path, vmin, vmax, fps=10):
    print(f"\n--- Creating GIF: {os.path.basename(save_path)} ---")
    _, W, T = frames1.shape
    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        for i in tqdm(range(T), desc="  - GIF progress"):
            f1_norm = cv2.normalize(np.clip(
                frames1[:, :, i], vmin, vmax), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            f2_norm = cv2.normalize(np.clip(
                frames2[:, :, i], vmin, vmax), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            f1_color, f2_color = cv2.cvtColor(
                f1_norm, cv2.COLOR_GRAY2BGR), cv2.cvtColor(f2_norm, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((f1_color, f2_color))
            cv2.putText(combined, label1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined, label2, (W + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.append_data(combined)
    print(f"  - Saved GIF to: {save_path}")


def save_final_leak_map(score_map, leaks, save_path):
    if score_map is None:
        return
    map_norm = cv2.normalize(score_map, None, 0, 255,
                             cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(map_norm, cv2.COLORMAP_HOT)
    for i, leak in enumerate(leaks):
        center_y, center_x = leak['coords']
        cv2.drawMarker(heatmap, (center_x, center_y), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(heatmap, f"#{i+1}", (center_x + 15, center_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(save_path, heatmap)
    print(f"  - Saved verification plot to: {os.path.basename(save_path)}")


def write_log_file(args, timings, slow_path_thread):
    # (Unchanged)
    log_path = os.path.join(args.output_dir, "run_log.txt")
    print(f"\n--- Writing execution log to: {log_path} ---")
    with open(log_path, 'w') as f:
        f.write("--- Leak Detection POC Execution Log ---\n\n")
        f.write("--- Run Parameters ---\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg:<25}: {value}\n")
        f.write("\n--- Performance Metrics ---\n")
        f.write(f"{'Total Execution Time':<25}: {timings['total']:.2f}s\n")
        f.write(f"{'Fast Path Loop Time':<25}: {timings['fast_path']:.2f}s\n")
        for regressor, result in slow_path_thread.results.items():
            f.write("\n" + "="*50 + "\n")
            f.write(f"--- Slow Path Results ({regressor.upper()}) ---\n")
            f.write(f"{'Computation Device':<25}: {result['device']}\n")
            f.write(f"{'Analysis Time':<25}: {result['time']:.2f}s\n")
            f.write(
                f"Safe Crop Window (HxW): {slow_path_thread.crop_dims[0]}x{slow_path_thread.crop_dims[1]}\n")
            if result['leaks']:
                f.write(f"Found {len(result['leaks'])} leak(s):\n")
                for i, leak in enumerate(result['leaks']):
                    f.write(
                        f"  - Leak #{i+1}: Coords={leak['coords']}, Score={leak['score']:.6f}\n")
            else:
                f.write("  - No leaks were detected.\n")
    print("  - Log file written successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Stabilization and Leak Detection POC.")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--analysis_window_frames", type=int, default=150)
    parser.add_argument("--num_leaks", type=int, default=2)
    parser.add_argument("--live_preview", action='store_true')
    parser.add_argument("--regressor", type=str,
                        default="theilsen", choices=["theilsen", "huber"])
    parser.add_argument("--compare", action='store_true')
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--shift1_frame", type=int, default=75)
    parser.add_argument("--shift1_x", type=float, default=-5.0)
    parser.add_argument("--shift1_y", type=float, default=0.0)
    parser.add_argument("--shift2_frame", type=int, default=150)
    parser.add_argument("--shift2_x", type=float, default=10.0)
    parser.add_argument("--shift2_y", type=float, default=0.0)
    args = parser.parse_args()

    total_start_time = time.perf_counter()
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.video_path))[0]

    try:
        all_frames = scipy.io.loadmat(args.video_path)[
            'TempFrames'].astype(np.float64)
        num_frames_to_process = min(
            all_frames.shape[2], args.analysis_window_frames)
        original_frames = all_frames[:, :, :num_frames_to_process]
        H, W, T = original_frames.shape
        print(
            f"--- Loaded video, processing {T} frames with shape: {original_frames.shape} ---")
    except Exception as e:
        sys.exit(f"FATAL: Could not load video file. Error: {e}")

    shaky_frames = add_gradual_shake(original_frames, args)
    reference_frame = original_frames[:, :, 0].astype(np.float32)

    ols_calculator = OnlineOLSCalculator(H, W)

    # Real-time camera implementation, need to set maxlen to 100-150 frames
    transform_history = deque(maxlen=T)
    slow_path_thread = SlowPathAnalyzer(
        args.num_leaks, T, args.regressor, args.gpu, args.compare)

    stabilized_frames_history = []
    live_preview_frames_history = []

    # list to store modifed frames for real-time simulation gif
    dynamic_crop_vis_history = []

    display_activity_map = np.zeros((H, W), dtype=np.float32)
    vmin, vmax = np.percentile(original_frames, (1, 99))

    print("--- Starting true real-time simulation loop... ---")
    slow_path_thread.start()
    main_loop_start_time = time.perf_counter()

    for i in tqdm(range(T), desc="Simulating real-time feed", file=sys.stdout):
        shaky_frame = shaky_frames[:, :, i]
        shift, _ = cv2.phaseCorrelate(
            reference_frame, shaky_frame.astype(np.float32))
        dx, dy = -shift[0], -shift[1]
        transform_history.append((dx, dy))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        stabilized_frame = cv2.warpAffine(shaky_frame, M, (W, H))
        stabilized_frames_history.append(stabilized_frame)

        dx_vals = [t[0] for t in transform_history]
        dy_vals = [t[1] for t in transform_history]
        y_start = math.ceil(max(dy_vals))
        y_end = H + math.floor(min(dy_vals))
        x_start = math.ceil(max(dx_vals))
        x_end = W + math.floor(min(dx_vals))

        mask = np.zeros((H, W), dtype=bool)
        mask[y_start:y_end, x_start:x_end] = True
        ols_calculator.add_frame(stabilized_frame, mask)
        slow_path_thread.add_frame(
            {'frame': stabilized_frame, 'transform': (dx, dy)})

        # to change the refresh rate of the live acitivy map(dependent on camera fps)
        if (i + 1) % 10 == 0 or i == T - 1:
            new_activity_map = ols_calculator.calculate_slope()
            display_activity_map = (
                0.2 * display_activity_map) + (0.8 * new_activity_map)

        if args.live_preview:
            cropped_map_for_display = display_activity_map[y_start:y_end, x_start:x_end]
            activity_display = np.zeros(
                (cropped_map_for_display.shape[0], cropped_map_for_display.shape[1], 3), dtype=np.uint8)
            if cropped_map_for_display.max() > 0:
                map_enhanced = cropped_map_for_display ** 1.5
                map_norm = cv2.normalize(
                    map_enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                activity_display = cv2.applyColorMap(
                    map_norm, cv2.COLORMAP_HOT)
            cv2.putText(activity_display, f"Frame: {i+1}/{T}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Live Activity Map (Fast Path)", activity_display)
            live_preview_frames_history.append(
                cv2.cvtColor(activity_display, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                slow_path_thread.stop()
                break

        # --- Create the frame for the new verification GIF ---
        vis_frame = cv2.normalize(np.clip(
            stabilized_frame, vmin, vmax), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
        # Draw the dynamic crop rectangle in bright green
        cv2.rectangle(vis_frame_bgr, (x_start, y_start),
                      (x_end, y_end), (0, 255, 0), 2)
        dynamic_crop_vis_history.append(
            cv2.cvtColor(vis_frame_bgr, cv2.COLOR_BGR2RGB))

    main_loop_end_time = time.perf_counter()
    print("\n--- Simulation finished. Waiting for background analysis... ---")
    slow_path_thread.join()
    if args.live_preview:
        cv2.destroyAllWindows()
    total_end_time = time.perf_counter()

    timings = {'total': total_end_time - total_start_time,
               'fast_path': main_loop_end_time - main_loop_start_time}

    stabilized_frames_np = np.stack(stabilized_frames_history, axis=2)
    gif_path = os.path.join(
        args.output_dir, f"{base_filename}_stabilization_comparison.gif")
    create_comparison_gif(shaky_frames, "1. Shaky",
                          stabilized_frames_np, "2. Stabilized", gif_path, vmin, vmax)

    if args.live_preview and live_preview_frames_history:
        live_gif_path = os.path.join(
            args.output_dir, f"{base_filename}_live_activity_map.gif")
        imageio.mimsave(live_gif_path, live_preview_frames_history, fps=15)
        print(f"\n--- Saved Live Preview GIF to: {live_gif_path} ---")

    if dynamic_crop_vis_history:
        print("\n--- Saving Dynamic Crop Visualization ---")
        crop_gif_path = os.path.join(
            args.output_dir, f"{base_filename}_dynamic_crop.gif")
        imageio.mimsave(crop_gif_path, dynamic_crop_vis_history, fps=15)
        print(f"  - Saved Dynamic Crop GIF to: {crop_gif_path}")

    print("\n--- Saving Final Leak Verification Plot(s) ---")
    for regressor, result in slow_path_thread.results.items():
        final_map_path = os.path.join(
            args.output_dir, f"{base_filename}_{regressor}_final_leak_map.png")
        save_final_leak_map(result['score_map'],
                            result['leaks'], final_map_path)

    write_log_file(args, timings, slow_path_thread)
    print("\n--- POC Finished Successfully ---")


if __name__ == "__main__":
    main()


"""
## CHPC
python src_cnn_v2/poc_two_stream_real_time_detection.py \
    --video_path "/scratch/general/vast/u1527145/datasets/Fluke_BrickCladding_2holes_0805_2025_noshutter/T1.6V_2025-08-05-19-57-24_20_34_14_.mat" \
    --output_dir "two_stream_output/chpc/Fluke_BrickCladding_2holes_0805_2025_noshutter/vid-1-left-right-frames50" \
    --num_leaks 2 \
    --shift1_frame 75 --shift1_x -5 --shift1_y 0 \
    --shift2_frame 150 --shift2_x 10 --shift2_y 0 \
    --analysis_window_frames 75 \
    --compare \
    --gpu


## LOCAL MACBOOK
python src_cnn_v2/poc_two_stream_real_time_detection.py \
    --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_HardyBoard_08132025_2holes_noshutter/T1.4V_2025-08-14-15-47-12_21_34_13_.mat" \
    --output_dir "two_stream_output/Fluke_HardyBoard_08132025_2holes_noshutter/vid-1-circular" \
    --num_leaks 2 \
    --analysis_window_frames 75 \
    --shift1_frame 25 --shift1_x -5 --shift1_y -10 \
    --shift2_frame 50 --shift2_x 10 --shift2_y 20 \
    --compare \
    --live_preview


# Gypsum
python src_cnn_v2/poc_two_stream_real_time_detection.py \
    --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_Gypsum_07162025_noshutter/T1.4V_2025-07-17-16-56-31_22_34_12_.mat" \
    --output_dir "two_stream_output/Fluke_Gypsum_07162025_noshutter/vid-1-left-right" \
    --num_leaks 1 \
    --shift1_frame 75 --shift1_x -5 --shift1_y 0 \
    --shift2_frame 150 --shift2_x 10 --shift2_y 0 \
    --live_preview

# Brickcladding
python src_cnn_v2/poc_two_stream_real_time_detection.py \
    --video_path "/Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0805_2025_noshutter/T1.6V_2025-08-05-19-57-24_20_34_14_.mat" \
    --output_dir "two_stream_output/Fluke_BrickCladding_2holes_0805_2025_noshutter/vid-1-left-right-frames50" \
    --num_leaks 2 \
    --shift1_frame 75 --shift1_x -5 --shift1_y 0 \
    --shift2_frame 150 --shift2_x 10 --shift2_y 0 \
    --analysis_window_frames 50 \
    --live_preview
"""
