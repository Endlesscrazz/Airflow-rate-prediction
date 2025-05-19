import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from matplotlib.patches import Rectangle # Not strictly used, but good for other bbox tasks
import os
import argparse

def get_mask_bounding_box(mask, padding=5):
    """
    Calculates the bounding box of the True values in a boolean mask.

    Args:
        mask (np.ndarray): The boolean mask.
        padding (int): Number of pixels to add as padding around the tight bounding box.

    Returns:
        tuple: (row_min, row_max, col_min, col_max) for the bounding box.
               Returns None if mask is empty or not a 2D array.
    """
    if mask is None or mask.ndim != 2 or not np.any(mask):
        print("Debug: Mask is None, not 2D, or all False in get_mask_bounding_box.")
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols): 
        print("Debug: No True values found along rows or columns in get_mask_bounding_box.")
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding
    rmin_padded = max(0, rmin - padding)
    cmin_padded = max(0, cmin - padding)
    # +1 because slicing is exclusive for the upper bound in Python
    rmax_padded = min(mask.shape[0], rmax + 1 + padding) 
    cmax_padded = min(mask.shape[1], cmax + 1 + padding)

    # Ensure rmin < rmax and cmin < cmax after padding
    if rmin_padded >= rmax_padded or cmin_padded >= cmax_padded:
        print(f"Debug: Invalid bbox after padding: r({rmin_padded}-{rmax_padded}), c({cmin_padded}-{cmax_padded}). Original r({rmin}-{rmax+1}), c({cmin}-{cmax+1}). Using original tight box.")
        # Fallback to tight box if padding makes it invalid (e.g., mask is very small)
        return int(rmin), int(rmax + 1), int(cmin), int(cmax + 1)

    return int(rmin_padded), int(rmax_padded), int(cmin_padded), int(cmax_padded)

def visualize_thermal_data(mat_file_path, mask_file_path, output_dir, 
                           mat_key='TempFrames', fps=5.0, later_frame_sec=3.0,
                           animation_duration_sec=10.0, animation_fps=10,
                           colormap='inferno', bbox_padding=10, output_format='mp4'):
    """
    Generates visualizations for thermal data:
    1. Static frame at t=0 with mask overlay.
    2. Static frame at a later time with mask overlay.
    3. Animation of the hotspot region over time (saved as MP4 or GIF).

    Args:
        mat_file_path (str): Path to the .mat file.
        mask_file_path (str): Path to the .npy mask file.
        output_dir (str): Directory to save visualizations.
        mat_key (str): Key for temperature frames in the .mat file.
        fps (float): Frames per second of the original thermal video.
        later_frame_sec (float): Time in seconds for the "later" static frame.
        animation_duration_sec (float): Duration of the hotspot animation in seconds.
        animation_fps (int): FPS for the output animated video.
        colormap (str): Colormap for thermal visualization.
        bbox_padding (int): Padding around the hotspot mask for the animation.
        output_format (str): 'mp4' or 'gif' for the animation.
    """
    # --- Create output directory if it doesn't exist ---
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(mat_file_path))[0]

    # --- Load Data ---
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        frames = mat_data.get(mat_key)
        if frames is None:
            print(f"Error: MAT key '{mat_key}' not found in {mat_file_path}.")
            return
        if frames.ndim != 3 or frames.shape[2] < 1:
            print(f"Error: Frames data in {mat_file_path} is not a valid 3D array.")
            return
        print(f"Loaded frames from {mat_file_path}, shape: {frames.shape}")
    except Exception as e:
        print(f"Error loading .mat file {mat_file_path}: {e}")
        return

    try:
        mask = np.load(mask_file_path)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.shape != frames.shape[:2]:
            print(f"Error: Mask shape {mask.shape} does not match frame shape {frames.shape[:2]}.")
            return
        print(f"Loaded mask from {mask_file_path}, shape: {mask.shape}, Non-zero elements: {np.sum(mask)}")
    except Exception as e:
        print(f"Error loading mask file {mask_file_path}: {e}")
        return

    H, W, num_total_frames = frames.shape

    # --- 1. Static Frame at t=0 ---
    frame0 = frames[:, :, 0]
    plt.figure(figsize=(10, 8)) 
    plt.imshow(frame0, cmap=colormap)
    plt.colorbar(label='Temperature (°C)')
    if np.any(mask):
        plt.contour(mask, colors='lime', linewidths=1.5, linestyles='solid') 
    plt.title(f'{base_filename} - Frame 0 (Initial State) with Mask Outline')
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    frame0_save_path = os.path.join(output_dir, f"{base_filename}_frame0_overlay.png")
    plt.savefig(frame0_save_path)
    plt.close()
    print(f"Saved initial frame visualization to {frame0_save_path}")

    # --- 2. Static Frame at Later Time ---
    later_frame_index = min(int(later_frame_sec * fps), num_total_frames - 1)
    if later_frame_index < 0: later_frame_index = 0 

    frame_later = frames[:, :, later_frame_index]
    plt.figure(figsize=(10, 8)) 
    plt.imshow(frame_later, cmap=colormap)
    plt.colorbar(label='Temperature (°C)')
    if np.any(mask): 
        plt.contour(mask, colors='lime', linewidths=1.5, linestyles='solid') 
    plt.title(f'{base_filename} - Frame at {later_frame_sec:.1f}s (Frame {later_frame_index}) with Mask Outline')
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    frame_later_save_path = os.path.join(output_dir, f"{base_filename}_frame_later_overlay.png")
    plt.savefig(frame_later_save_path)
    plt.close()
    print(f"Saved later frame visualization to {frame_later_save_path}")

    # --- 3. Animation of Hotspot Region ---
    if not np.any(mask):
        print("Mask is empty. Skipping hotspot animation.")
        return
        
    bbox = get_mask_bounding_box(mask, padding=bbox_padding)
    if bbox is None:
        print("Could not determine bounding box for the mask. Skipping animation.")
        return

    rmin, rmax, cmin, cmax = bbox
    
    if rmin >= rmax or cmin >= cmax:
        print(f"Invalid bounding box: r({rmin}-{rmax}), c({cmin}-{cmax}). Skipping animation.")
        return

    print(f"Hotspot bounding box for animation (r_min, r_max, c_min, c_max): ({rmin}, {rmax}, {cmin}, {cmax})")

    num_animation_frames = min(int(animation_duration_sec * fps), num_total_frames)
    if num_animation_frames <= 0:
        print("Not enough frames for animation based on duration. Skipping animation.")
        return

    hotspot_frames_for_anim = []
    for i in range(num_animation_frames):
        frame_data = frames[:, :, i]
        hotspot_region = frame_data[rmin:rmax, cmin:cmax]
        if hotspot_region.size == 0: 
            print(f"Warning: Empty hotspot region extracted at frame {i}. Using placeholder.")
            expected_rows = max(1, rmax - rmin)
            expected_cols = max(1, cmax - cmin)
            hotspot_frames_for_anim.append(np.full((expected_rows, expected_cols), np.nan))
        else:
            hotspot_frames_for_anim.append(hotspot_region)
            
    if not hotspot_frames_for_anim:
        print("No valid hotspot frames extracted for animation. Skipping.")
        return

    valid_hotspot_data = [hf.flatten() for hf in hotspot_frames_for_anim if hf.size > 0 and np.any(np.isfinite(hf))]
    if not valid_hotspot_data:
        print("No valid pixel data in hotspot regions for animation. Skipping.")
        return
    all_hotspot_pixels = np.concatenate(valid_hotspot_data)
    
    if all_hotspot_pixels.size == 0:
        print("Concatenated hotspot pixels array is empty. Skipping animation.")
        return
        
    vmin = np.nanmin(all_hotspot_pixels)
    vmax = np.nanmax(all_hotspot_pixels)
    
    if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax: 
        print("Warning: Could not determine dynamic color scale for animation. Using fallback.")
        first_hotspot_frame_pixels = hotspot_frames_for_anim[0].flatten()
        if first_hotspot_frame_pixels.size > 0 and np.any(np.isfinite(first_hotspot_frame_pixels)):
            vmin = np.nanmin(first_hotspot_frame_pixels); vmax = np.nanmax(first_hotspot_frame_pixels)
            if vmin == vmax: vmin -= 0.5; vmax += 0.5
        else: vmin = 20; vmax = 30


    fig_anim, ax_anim = plt.subplots(figsize=(7,6))
    
    first_valid_hs_frame_for_imshow = next((hf for hf in hotspot_frames_for_anim if hf.size > 0 and np.any(np.isfinite(hf))), 
                                           np.zeros((max(1,rmax-rmin), max(1,cmax-cmin))))
                                           
    im = ax_anim.imshow(first_valid_hs_frame_for_imshow, cmap=colormap, vmin=vmin, vmax=vmax, 
                        interpolation='nearest') 
    
    ax_anim.set_title(f'Hotspot Temp Evolution ({base_filename})\nRegion: R[{rmin}:{rmax-1}], C[{cmin}:{cmax-1}]')
    ax_anim.set_xlabel('Pixel Column (Relative to Hotspot)')
    ax_anim.set_ylabel('Pixel Row (Relative to Hotspot)')
    
    cbar = fig_anim.colorbar(im, ax=ax_anim, label='Temperature (°C)', fraction=0.046, pad=0.04)

    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, color='white', 
                             fontsize=10, bbox=dict(facecolor='black', alpha=0.7))

    def update_anim(frame_idx_anim):
        if frame_idx_anim < len(hotspot_frames_for_anim):
            hs_frame_data = hotspot_frames_for_anim[frame_idx_anim]
            if hs_frame_data.size > 0:
                 im.set_data(hs_frame_data)
            current_time_sec = frame_idx_anim / fps 
            time_text.set_text(f'Time: {current_time_sec:.2f}s')
        return [im, time_text]

    anim = animation.FuncAnimation(fig_anim, update_anim, frames=num_animation_frames,
                                   interval=max(20, 1000//animation_fps), blit=True, repeat=False)

    if output_format.lower() == 'mp4':
        animation_save_path = os.path.join(output_dir, f"{base_filename}_hotspot_animation.mp4")
        writer = 'ffmpeg'
        extra_args = ['-vcodec', 'libx264'] # Common codec for MP4
    elif output_format.lower() == 'gif':
        animation_save_path = os.path.join(output_dir, f"{base_filename}_hotspot_animation.gif")
        writer = 'pillow'
        extra_args = None
    else:
        print(f"Unsupported output format: {output_format}. Defaulting to GIF.")
        animation_save_path = os.path.join(output_dir, f"{base_filename}_hotspot_animation.gif")
        writer = 'pillow'
        extra_args = None

    try:
        if extra_args:
             anim.save(animation_save_path, writer=writer, fps=animation_fps, extra_args=extra_args)
        else:
             anim.save(animation_save_path, writer=writer, fps=animation_fps)
        print(f"Saved hotspot animation to {animation_save_path} (Format: {output_format.upper()})")
        if output_format.lower() == 'mp4':
            print("NOTE: MP4 saving requires FFmpeg to be installed and in your system's PATH.")
    except Exception as e:
        print(f"Error saving animation as {output_format.upper()}: {e}")
        if output_format.lower() == 'mp4':
            print("Ensure FFmpeg is installed and accessible. You might need to specify the full path to ffmpeg.exe "
                  "in matplotlib.rcParams['animation.ffmpeg_path'] = 'C:/path/to/ffmpeg.exe'")
        elif output_format.lower() == 'gif':
             print("Ensure 'pillow' is installed (`pip install Pillow`).")
    
    plt.close(fig_anim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize thermal data and hotspot evolution.")
    parser.add_argument("mat_file", help="Path to the .mat file containing thermal frames.")
    parser.add_argument("mask_file", help="Path to the .npy file containing the hotspot mask.")
    parser.add_argument("output_dir", help="Directory to save the output visualizations.")
    
    parser.add_argument("--mat_key", default="TempFrames", help="Key for temperature frames in .mat file.")
    parser.add_argument("--fps", type=float, default=5.0, help="FPS of the original thermal video (for time calculation).")
    parser.add_argument("--later_frame_sec", type=float, default=15.0, help="Time for the 'later' static frame.")
    parser.add_argument("--animation_duration_sec", type=float, default=10.0, help="Duration of the hotspot animation.")
    parser.add_argument("--animation_fps", type=int, default=10, help="Playback FPS for the output animation.")
    parser.add_argument("--colormap", default="inferno", help="Colormap for thermal visualization.")
    parser.add_argument("--bbox_padding", type=int, default=10, help="Padding around hotspot mask for animation.")
    parser.add_argument("--output_format", default="mp4", choices=['mp4', 'gif'], help="Output format for the animation (mp4 or gif). Default: mp4")


    args = parser.parse_args()
    
    if not os.path.isfile(args.mat_file):
        print(f"Error: MAT file not found at {args.mat_file}")
        exit()
    if not os.path.isfile(args.mask_file):
        print(f"Error: Mask file not found at {args.mask_file}")
        exit()

    visualize_thermal_data(
        mat_file_path=args.mat_file,
        mask_file_path=args.mask_file,
        output_dir=args.output_dir,
        mat_key=args.mat_key,
        fps=args.fps,
        later_frame_sec=args.later_frame_sec,
        animation_duration_sec=args.animation_duration_sec,
        animation_fps=args.animation_fps,
        colormap=args.colormap,
        bbox_padding=args.bbox_padding,
        output_format=args.output_format
    )
