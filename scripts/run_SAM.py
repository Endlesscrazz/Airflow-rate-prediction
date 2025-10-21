# run_SAM.py
"""
FINAL UNIVERSAL BATCH-PROCESSING SCRIPT (v10 - With Debugging & Correct Coords)
- Includes coordinate-based assignment for 2-hole and 10-hole datasets.
- Uses the verified, correct coordinates for robust hole ID assignment.
- Adds a '--debug' flag for verbose path and logic tracing.
- Generates a '_prompt_assignment.png' plot for visual verification in assignment modes.
"""

import glob
import os
import sys
import argparse
import numpy as np
import scipy.io
import cv2
import torch
import matplotlib.pyplot as plt
import fnmatch
from scipy.stats import kendalltau, mstats
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import traceback

try:
    from segment_anything import sam_model_registry, SamPredictor
    from skimage.feature import peak_local_max
except ImportError:
    print("Error: Required libraries not found. Please run:", file=sys.stderr)
    print("pip install scikit-image segment-anything-py imageio matplotlib", file=sys.stderr)
    sys.exit(1)

# --- Coordinates for Hole Assignment (Verified) ---
MANUAL_PROMPTS = {
    "Fluke_BrickCladding_2holes_0808_2025_noshutter": [
        {"name": "1", "coord": (272, 326), "box_size": 15},  # center hole
        {"name": "2", "coord": (360, 140), "box_size": 15}  # bottom_left hole
    ],
    "Fluke_BrickCladding_2holes_0805_2025_noshutter": [
        {"name": "1", "coord": (274, 328), "box_size": 15},  # center hole
        {"name": "2", "coord": (360, 140), "box_size": 15}  # bottom_left hole
    ],
    "Fluke_BrickCladding_2holes_0616_2025_noshutter": [
        {"name": "1", "coord": (302, 332), "box_size": 30},  # center hole
        # top left corner hole
        {"name": "2", "coord": (128, 251), "box_size": 15}
    ]

}


# 10-hole Gypsum dataset
SLIT_COORDS = {'Hole_1': np.array([233, 553]), 'Hole_10': np.array([414, 353])}
TARGET_HOLE_COORDS = {
    2: np.array([79, 507]), 3: np.array([53, 344]), 4: np.array([78, 181]),
    5: np.array([213, 219]), 6: np.array([332, 149]), 7: np.array([326, 307]),
    8: np.array([325, 453]), 9: np.array([201, 355]),
}
# 2-hole datasets (y, x)
TWO_HOLE_COORDS = {
    'hardyboard': {
        1: np.array([322, 328]),  # center hole (from 0813 dataset)
        2: np.array([131, 497])  # right corner hole (from 0813 dataset)
    },
}

# --- Helper & Core Functions ---


def save_parameters(args, output_dir):
    params_path = os.path.join(output_dir, "parameters.txt")
    with open(params_path, 'w') as f:
        f.write("--- Run Parameters ---\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")


def apply_spatial_filter(frames, filter_type='none'):
    if filter_type.lower() == 'none':
        return frames
    H, W, T = frames.shape
    filtered_frames = np.zeros_like(frames, dtype=np.float64)
    for i in range(T):
        frame_8bit = cv2.normalize(
            frames[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if filter_type.lower() == 'bilateral':
            filtered_frame_8bit = cv2.bilateralFilter(frame_8bit, 9, 75, 75)
        else:
            filtered_frame_8bit = frame_8bit
        min_val, max_val = np.min(frames[:, :, i]), np.max(frames[:, :, i])
        filtered_frames[:, :, i] = cv2.normalize(filtered_frame_8bit.astype(
            np.float64), None, min_val, max_val, cv2.NORM_MINMAX)
    return filtered_frames


def apply_temporal_smoothing(frames, window_size):
    if window_size <= 1:
        return frames
    H, W, T = frames.shape
    smoothed_frames = np.zeros_like(frames, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    for r in range(H):
        for c in range(W):
            smoothed_series = np.convolve(
                frames[r, c, :], kernel, mode='valid')
            pad_before = (T - len(smoothed_series)) // 2
            smoothed_frames[r, c, pad_before:pad_before +
                            len(smoothed_series)] = smoothed_series
    return smoothed_frames


def _calculate_slope_for_row(row_data, t, method):
    W = row_data.shape[0]
    row_values = np.zeros(W, dtype=np.float64)
    for c in range(W):
        pixel_series = row_data[c, :]
        if not np.any(np.isnan(pixel_series)) and len(pixel_series) > 1:
            try:
                if method == 'theil_sen':
                    val, _, _, _ = mstats.theilslopes(pixel_series, t, 0.95)
                else:
                    val, _ = kendalltau(t, pixel_series)
                row_values[c] = val if np.isfinite(val) else 0.0
            except (ValueError, IndexError):
                pass
    return row_values


def generate_activity_map(frames, method, env_para):
    H, W, T = frames.shape
    t = np.arange(T)
    if T < 2:
        return np.zeros((H, W), dtype=np.float64)
    results = Parallel(n_jobs=-1)(delayed(_calculate_slope_for_row)
                                  (frames[r, :, :], t, method) for r in range(H))
    raw_activity_map = np.vstack(results)
    if env_para == 1:
        activity_map = raw_activity_map.copy()
        activity_map[activity_map < 0] = 0
    elif env_para == -1:
        activity_map = -raw_activity_map
        activity_map[activity_map < 0] = 0
    else:
        activity_map = np.abs(raw_activity_map)
    return activity_map


def create_border_roi_mask(frame_shape, border_percent):
    H, W = frame_shape
    border_h, border_w = int(H * border_percent), int(W * border_percent)
    roi_mask = np.zeros(frame_shape, dtype=np.uint8)
    roi_mask[border_h:H-border_h, border_w:W-border_w] = 1
    return roi_mask


def find_prompts_manually(manual_prompts_db, dataset_key, frame_shape):
    """Looks up and prepares prompts from a hardcoded dictionary."""
    leaks_to_use = None
    for key in manual_prompts_db:
        if key in dataset_key:
            leaks_to_use = manual_prompts_db[key]
            break
    if not leaks_to_use:
        return []

    prompts = []
    for leak_def in leaks_to_use:
        r, c = leak_def["coord"]

        cand = {
            'centroid': np.array([c, r]),
            'peak_activity': 999,
            'hole_id': leak_def.get("name"),
            'box_size': leak_def.get("box_size")
        }
        prompts.append(cand)
    return prompts


def find_all_local_peaks(activity_map, min_distance, abs_threshold):
    coordinates = peak_local_max(
        activity_map, min_distance=min_distance, threshold_abs=abs_threshold)
    if coordinates.size == 0:
        return []
    all_candidates = [{'centroid': np.array(
        [c, r]), 'peak_activity': activity_map[r, c]} for r, c in coordinates]
    return sorted(all_candidates, key=lambda x: x['peak_activity'], reverse=True)


def filter_and_assign_prompts(all_candidates, slit_coords_to_filter, target_hole_coords, exclusion_radius=30):
    filtered_prompts = []
    if slit_coords_to_filter:
        slit_locations = np.array([coord[::-1]
                                  for coord in slit_coords_to_filter.values()])
        for cand in all_candidates:
            if np.all(cdist(cand['centroid'].reshape(1, 2), slit_locations) > exclusion_radius):
                filtered_prompts.append(cand)
    else:
        filtered_prompts = all_candidates

    if not filtered_prompts:
        return []

    assigned_prompts, remaining_prompts = {}, list(filtered_prompts)
    for hole_id, target_coord in target_hole_coords.items():
        if not remaining_prompts:
            break
        target_xy = target_coord[::-1]
        distances = [np.linalg.norm(p['centroid'] - target_xy)
                     for p in remaining_prompts]
        closest_prompt = remaining_prompts.pop(np.argmin(distances))
        closest_prompt['hole_id'] = hole_id
        assigned_prompts[hole_id] = closest_prompt

    return [assigned_prompts[key] for key in sorted(assigned_prompts.keys())]


def visualize_manual_prompts(base_frame_rgb, final_prompts, save_path):
    """Creates a debug plot showing the location of manually defined prompts."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(cv2.cvtColor(base_frame_rgb, cv2.COLOR_BGR2RGB))
    ax.set_title('Debug Map: Manual Prompts')

    if final_prompts:
        ax.scatter([p['centroid'][0] for p in final_prompts], [p['centroid'][1] for p in final_prompts],
                   s=600, c='lime', marker='*', edgecolor='black', zorder=5, label='Manual Prompts')
        for p in final_prompts:
            ax.text(p['centroid'][0] + 15, p['centroid'][1] + 15,
                    str(p['hole_id']), color='white', fontsize=16, fontweight='bold')

    ax.legend()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  - Saved manual prompt visual to: {os.path.basename(save_path)}")


def visualize_assigned_prompts(activity_map, all_candidates, final_prompts, save_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(activity_map, cmap='hot', vmin=0)
    fig.colorbar(im, ax=ax, label='Activity Map')
    ax.set_title('Debug Map: Prompts Assigned & Labeled')
    if all_candidates:
        ax.scatter([c['centroid'][0] for c in all_candidates], [c['centroid'][1] for c in all_candidates],
                   s=150, facecolors='none', edgecolors='cyan', lw=1.5, label='All Found Peaks')
    if final_prompts:
        ax.scatter([p['centroid'][0] for p in final_prompts], [p['centroid'][1] for p in final_prompts],
                   s=600, c='yellow', marker='*', edgecolor='black', zorder=5, label='Assigned Prompts')
        for p in final_prompts:
            ax.text(p['centroid'][0] + 15, p['centroid'][1] + 15,
                    str(p['hole_id']), color='white', fontsize=16, fontweight='bold')
    ax.legend()
    plt.savefig(save_path)
    plt.close(fig)
    print(
        f"  - Saved prompt assignment visual to: {os.path.basename(save_path)}")


def run_sam_with_box_and_point_prompts(frame_rgb, top_candidates, predictor, default_prompt_box_size):
    """
    Runs SAM. Now checks for a custom 'box_size' on each candidate before
    falling back to the default command-line size.
    """
    all_final_masks = []
    predictor.set_image(frame_rgb)
    H, W, _ = frame_rgb.shape

    for cand in top_candidates:
        center_x, center_y = cand['centroid']

        box_size_to_use = cand.get('box_size', default_prompt_box_size)

        box_half = box_size_to_use // 2

        x1, y1 = int(np.clip(center_x - box_half, 0, W-1)
                     ), int(np.clip(center_y - box_half, 0, H-1))
        x2, y2 = int(np.clip(center_x + box_half, 0, W-1)
                     ), int(np.clip(center_y + box_half, 0, H-1))

        point_coords, point_labels = cand['centroid'].reshape(
            1, 2), np.array([1])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=np.array([
                                        [x1, y1, x2, y2]]), multimask_output=False)
        if len(masks) > 0:
            all_final_masks.append(masks[0])

    return all_final_masks

# --- Main Execution ---


def main(args):
    print("--- Starting Universal Hotspot Segmentation Batch Process ---")
    if args.debug:
        print("\n*** DEBUG MODE ENABLED ***\n")

    os.makedirs(args.base_output_dir, exist_ok=True)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    search_pattern = os.path.join(args.dataset_dir, '**', '*.mat')
    mat_file_paths = [p for p in glob.glob(
        search_pattern, recursive=True) if not os.path.basename(p).startswith('._')]
    if not mat_file_paths:
        print("Warning: No .mat files found. Exiting.")
        return

    iterator = mat_file_paths
    if not args.debug:
        iterator = tqdm(mat_file_paths, desc="Processing Videos")

    for video_path in iterator:
        try:
            video_filename = os.path.basename(video_path)
            base_filename = os.path.splitext(video_filename)[0]
            relative_path = os.path.relpath(
                os.path.dirname(video_path), args.dataset_dir)
            video_output_dir = os.path.join(
                args.base_output_dir, relative_path, base_filename)
            os.makedirs(video_output_dir, exist_ok=True)
            save_parameters(args, video_output_dir)

            if args.debug:
                print(f"\n--- DEBUG: Processing video: {video_path} ---")

            frames = scipy.io.loadmat(video_path)[
                "TempFrames"].astype(np.float64)
            H, W, T = frames.shape
            if T < 2:
                raise ValueError("Not enough frames.")
            target_frame = np.median(frames, axis=2)

            final_prompts = []

            if args.processing_mode == 'manual':
                print("\n  - Running in 'manual' mode.")
                dataset_key = os.path.basename(
                    os.path.normpath(args.dataset_dir))
                final_prompts = find_prompts_manually(
                    MANUAL_PROMPTS, dataset_key, (H, W))

                if final_prompts:
                    frame_rgb_for_plot = cv2.cvtColor(cv2.normalize(
                        target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_GRAY2BGR)
                    verification_path = os.path.join(
                        video_output_dir, f"{base_filename}_prompt_verification.png")
                    visualize_manual_prompts(
                        frame_rgb_for_plot, final_prompts, verification_path)
            else:
                temporally_smoothed_frames = apply_temporal_smoothing(
                    apply_spatial_filter(frames, args.spatial_filter), args.temporal_smooth_window)
                activity_map = generate_activity_map(
                    temporally_smoothed_frames, args.activity_method, args.env_para)
                if args.roi_method == 'border':
                    roi_mask = create_border_roi_mask(
                        (H, W), args.roi_border_percent)

                masked_activity_map = activity_map * roi_mask

                all_candidates = find_all_local_peaks(
                    masked_activity_map, args.peak_min_distance, args.peak_abs_threshold)

                if args.processing_mode == 'multi_hole_filter':
                    final_prompts = filter_and_assign_prompts(
                        all_candidates, SLIT_COORDS, TARGET_HOLE_COORDS)
                    visualize_assigned_prompts(masked_activity_map, all_candidates, final_prompts, os.path.join(
                        video_output_dir, f"{base_filename}_prompt_assignment.png"))
                elif args.processing_mode == 'two_hole_assign':
                    material_coords, input_path_lower = None, video_path.lower().replace("_",
                                                                                         "").replace("-", "")
                    for material, coords in TWO_HOLE_COORDS.items():
                        if material in input_path_lower:
                            material_coords = coords
                            break
                    if material_coords:
                        final_prompts = filter_and_assign_prompts(
                            all_candidates, {}, material_coords)
                        visualize_assigned_prompts(masked_activity_map, all_candidates, final_prompts, os.path.join(
                            video_output_dir, f"{base_filename}_prompt_assignment.png"))
                    else:
                        print(
                            f"    - Warning: No coordinates found for this material. Skipping assignment for {base_filename}.", file=sys.stderr)
                        continue
                else:  # standard
                    final_prompts = all_candidates[:args.num_leaks]
                    for i, p in enumerate(final_prompts):
                        p['hole_id'] = i + 1

            if not final_prompts:
                print(
                    f"  No prompts found for {video_filename}. Skipping.", file=sys.stderr)
                continue

            frame_rgb = cv2.cvtColor(cv2.normalize(
                target_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_GRAY2BGR)
            all_final_masks = run_sam_with_box_and_point_prompts(
                frame_rgb, final_prompts, predictor, default_prompt_box_size=args.prompt_box_size)
            if not all_final_masks:
                print(
                    f"  SAM generated no masks for {video_filename}. Skipping.", file=sys.stderr)
                continue

            segmentation_image = frame_rgb.copy()
            colors = [[255, 255, 0], [0, 255, 255]]
            for i, mask in enumerate(all_final_masks):
                hole_id = final_prompts[i].get('hole_id')
                np.save(os.path.join(video_output_dir,
                        f"{base_filename}_mask_{hole_id}.npy"), mask)
                color_overlay = np.zeros_like(segmentation_image)
                color_overlay[mask] = colors[i % len(colors)]
                segmentation_image = cv2.addWeighted(
                    segmentation_image, 1, color_overlay, 0.6, 0)

            cv2.imwrite(os.path.join(
                video_output_dir, f"{base_filename}_sam_segmentation.png"), segmentation_image)
            print(
                f"  - Saved {len(all_final_masks)} masks and visuals to {video_output_dir}")

        except Exception as e:
            print(f"\n--- ERROR processing {video_filename} ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            if args.debug:
                traceback.print_exc()
                sys.exit(1)
            continue

    print("\n--- Batch Process Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universal batch-processing script for hotspot segmentation.")
    parser.add_argument("--debug", action='store_true',
                        help="Enable verbose debugging output and stop on the first error.")
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--base_output_dir", required=True, type=str)
    default_checkpoint = os.path.join(
        'SAM', 'sam_checkpoints', 'sam_vit_b_01ec64.pth')
    parser.add_argument("--checkpoint_path", type=str,
                        default=default_checkpoint)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--processing_mode", type=str, default="standard",
                        choices=['standard', 'two_hole_assign', 'multi_hole_filter', 'manual'])

    param_group = parser.add_argument_group('Pipeline Control Parameters')
    param_group.add_argument("--num_leaks", type=int, default=1)
    param_group.add_argument("--env_para", type=int,
                             default=1, choices=[1, -1, 0])
    param_group.add_argument("--activity_method", type=str,
                             default="kendall_tau", choices=['theil_sen', 'kendall_tau'])
    param_group.add_argument("--temporal_smooth_window", type=int, default=3)
    param_group.add_argument("--spatial_filter", type=str,
                             default="bilateral", choices=['none', 'bilateral'])
    param_group.add_argument("--roi_method", type=str,
                             default="border", choices=['otsu', 'border'])
    param_group.add_argument("--roi_border_percent", type=float, default=0.1)
    param_group.add_argument("--roi_erosion", type=int, default=3)

    prompt_group = parser.add_argument_group('Prompt Finding Method')
    prompt_group.add_argument("--peak_min_distance", type=int, default=50)
    prompt_group.add_argument("--peak_abs_threshold", type=float, default=0.2)
    prompt_group.add_argument("--prompt_box_size", type=int, default=30)

    args = parser.parse_args()
    main(args)
"""
python -m scripts.run_SAM --dataset_dir /Volumes/One_Touch/Airflow-rate-prediction/datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter \
    --base_output_dir output_SAM/datasets/Fluke_BrickCladding_2holes_0616_2025_noshutter \
    --num_leaks 2 \
    --roi_method border \
    --processing_mode manual \
    --prompt_box_size 15
    
"""
