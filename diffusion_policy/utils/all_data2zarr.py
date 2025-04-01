import os
import glob
import pickle
import numpy as np
import zarr
from tqdm import tqdm
from PIL import Image
import json
from numcodecs import Blosc

try:
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
    register_codecs()
except ImportError:
    print("Warning: Jpeg2k codec not available, falling back to Blosc")
    Jpeg2k = None

def convert_custom_to_zarr(input_path, output_zarr_path, shape_meta, env_meta):
    # Initialize Zarr store as a directory
    root = zarr.open(output_zarr_path, mode='w')
    data_group = root.require_group('data')
    meta_group = root.require_group('meta')

    # Extract metadata
    obs_shape_meta = shape_meta['obs']
    rgb_keys = [k for k, v in obs_shape_meta.items() if v.get('type', 'low_dim') == 'rgb']
    lowdim_keys = [k for k, v in obs_shape_meta.items() if v.get('type', 'low_dim') == 'low_dim']

    # Get all demo directories (all subdirectories under input_path)
    demo_dirs = sorted([d for d in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(d)])
    if not demo_dirs:
        raise ValueError(f"No episode directories found in {input_path}")

    print(f"Found {len(demo_dirs)} episode directories: {demo_dirs}")

    episode_ends = []
    episode_starts = [0]
    all_data = {key: [] for key in lowdim_keys + ['action']}
    all_images = {key: [] for key in rgb_keys}

    total_steps = 0
    # Process each demo directory
    for demo_dir in tqdm(demo_dirs, desc="Processing demo folders"):
        # Get all .pkl files and deduplicate by filename
        pkl_files = sorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
        # Extract unique filenames (ignoring duplicates)
        unique_pkl_files = sorted(list(set(os.path.basename(f) for f in pkl_files)))
        # Reconstruct full paths with unique filenames
        pkl_files = [os.path.join(demo_dir, f) for f in unique_pkl_files]

        episode_length = len(pkl_files)
        if episode_length == 0:
            print(f"Warning: No .pkl files found in {demo_dir}, skipping.")
            continue

        total_steps += episode_length
        episode_ends.append(episode_starts[-1] + episode_length)
        if episode_ends[-1] != episode_starts[-1]:
            episode_starts.append(episode_ends[-1])

        for pkl_idx, pkl_file in enumerate(pkl_files):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # Collect lowdim data
            for key in lowdim_keys:
                if key in data:
                    value = np.array(data[key], dtype=np.float32)
                    if value.shape != shape_meta['obs'][key]['shape']:
                        raise ValueError(f"Shape mismatch for {key} in {pkl_file}: "
                                       f"got {value.shape}, expected {shape_meta['obs'][key]['shape']}")
                    all_data[key].append(value)
                else:
                    raise KeyError(f"Required key {key} not found in {pkl_file}")

            # Collect action (mapped from joint_positions)
            if 'joint_positions_dummy' in data:
                value = np.array(data['joint_positions_dummy'], dtype=np.float32)
                expected_action_shape = shape_meta['action']['joint_positions_dummy']['shape']
                if value.shape != expected_action_shape:
                    raise ValueError(f"Shape mismatch for action in {pkl_file}: "
                                   f"got {value.shape}, expected {expected_action_shape}")
                all_data['action'].append(value)
            else:
                raise KeyError(f"Required key 'joint_positions' not found in {pkl_file}")

            # Collect and resize images (view_5 for image_1, view_254 for image_2)
            for cam_idx, rgb_key in enumerate(rgb_keys, 1):
                cam_dir = 'view_5' if cam_idx == 1 else 'view_254'  # image_1 -> view_5, image_2 -> view_254
                img_path = os.path.join(demo_dir, cam_dir, f'{pkl_idx:05d}.png')
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_resized = img.resize((320, 240), Image.Resampling.LANCZOS)  # (width, height)
                    img_array = np.array(img_resized)
                    if img_array.ndim == 2:  # Grayscale
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[-1] == 4:  # RGBA
                        img_array = img_array[..., :3]
                    img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                    expected_shape = shape_meta['obs'][rgb_key]['shape']  # (3, 240, 320)
                    if img_array.shape != expected_shape:
                        raise ValueError(f"Image shape mismatch for {rgb_key} in {img_path}: "
                                       f"got {img_array.shape}, expected {expected_shape}")
                    all_images[rgb_key].append(img_array)
                else:
                    raise FileNotFoundError(f"Image not found: {img_path}")

    n_steps = episode_ends[-1]
    if n_steps != total_steps:
        raise ValueError(f"Calculated steps {n_steps} doesn't match total steps {total_steps}")

    # Save episode ends
    meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None)

    # Save lowdim data
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    for key in tqdm(lowdim_keys + ['action'], desc="Saving lowdim data"):
        this_data = np.stack(all_data[key], axis=0)
        print(f"Saving {key} with shape: {this_data.shape}")
        expected_shape = (n_steps,) + (shape_meta['action']['joint_positions_dummy']['shape'] if key == 'action' 
                                     else shape_meta['obs'][key]['shape'])
        if this_data.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {key}: got {this_data.shape}, expected {expected_shape}")
        data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=(1,) + (shape_meta['obs'][key]['shape'] if key != 'action' 
                          else shape_meta['action']['joint_positions_dummy']['shape']),
            compressor=compressor,
            dtype=np.float32
        )

    # Save image data
    image_compressor = Jpeg2k(level=50) if Jpeg2k is not None else Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    for key in tqdm(rgb_keys, desc="Saving image data"):
        img_data = np.stack(all_images[key], axis=0)
        print(f"Saving {key} with shape: {img_data.shape}")
        if img_data.shape[0] != n_steps:
            raise ValueError(f"Number of images ({img_data.shape[0]}) doesn't match steps ({n_steps})")
        data_group.array(
            name=key,
            data=img_data,
            shape=img_data.shape,
            chunks=(1,) + shape_meta['obs'][key]['shape'],
            compressor=image_compressor,
            dtype=np.uint8
        )

    # Save environment metadata
    data_group.attrs['env_args'] = json.dumps(env_meta)

    print(f"Zarr directory saved to {output_zarr_path}")

    # Verify saved data
    root = zarr.open(output_zarr_path, mode='r')
    for key in ['joint_positions', 'joint_positions_dummy', 'state', 'image_1', 'image_2', 'action']:
        arr = root['data'][key]
        print(f"Loaded {key} shape: {arr.shape}")
        if key in ['joint_positions', 'joint_positions_dummy', 'state', 'action']:
            print(f"Loaded {key} first 2 timesteps: {arr[:2]}")

if __name__ == "__main__":
    shape_meta = {
        'obs': {
            'joint_positions': {'shape': (7,), 'type': 'low_dim'},
            'joint_positions_dummy': {'shape': (19,), 'type': 'low_dim'},
            'state': {'shape': (7,), 'type': 'low_dim'},
            'image_1': {'shape': (3, 240, 320), 'type': 'rgb'},  # (C, H, W) from view_5
            'image_2': {'shape': (3, 240, 320), 'type': 'rgb'},  # (C, H, W) from view_254
        },
        'action': {
            'joint_positions_dummy': {'shape': (19,)}
        }
    }

    env_meta = {
        'env_name': 'custom_robot_env',
        'type': 'robomimic',
        'env_kwargs': {}
    }

    input_path = "/shared_disk/datasets/public_datasets/SplatSim/bc_data_view/gello"
    output_zarr_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello_all_dummy.zarr"

    convert_custom_to_zarr(input_path, output_zarr_path, shape_meta, env_meta)