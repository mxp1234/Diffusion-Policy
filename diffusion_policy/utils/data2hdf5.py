import os
import glob
import pickle
import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image
import json

def convert_custom_to_hdf5(input_path, hdf5_path, shape_meta, env_meta, target_size=(240, 320)):
    # Open HDF5 file for writing
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        # Create root groups
        data_group = hdf5_file.create_group('data')
        data_group.attrs['env_args'] = json.dumps(env_meta)

        # Extract observation and action metadata
        obs_shape_meta = shape_meta['obs']
        rgb_keys = [k for k, v in obs_shape_meta.items() if v.get('type', 'low_dim') == 'rgb']
        lowdim_keys = [k for k, v in obs_shape_meta.items() if v.get('type', 'low_dim') == 'low_dim']
        action_key = 'action'

        # Get demo directories (episodes)
        demo_dirs = sorted(glob.glob(os.path.join(input_path, "[0-9]")))
        n_episodes = len(demo_dirs)

        # Process each demo (episode)
        for i, demo_dir in enumerate(tqdm(demo_dirs, desc="Processing episodes")):
            pkl_files = sorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
            episode_len = len(pkl_files)

            # Temporary storage for episode data
            all_data = {key: [] for key in lowdim_keys + [action_key]}
            all_images = {key: [] for key in rgb_keys}

            # Process each timestep in the episode
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

                # Collect action
                if action_key in data:
                    value = np.array(data[action_key], dtype=np.float32)
                    if value.shape != shape_meta[action_key]['shape']:
                        raise ValueError(f"Shape mismatch for action in {pkl_file}: "
                                       f"got {value.shape}, expected {shape_meta[action_key]['shape']}")
                    all_data[action_key].append(value)
                else:
                    raise KeyError(f"Required key 'action' not found in {pkl_file}")

                # Collect and resize images
                for cam_idx, rgb_key in enumerate(rgb_keys, 1):
                    img_path = os.path.join(demo_dir, f'images_{cam_idx}', f'{pkl_idx:05d}.png')
                    if os.path.exists(img_path):
                        # Open image and resize
                        img = Image.open(img_path).resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                        img_array = np.array(img)  # Shape: (H, W, C)

                        # Ensure 3 channels (RGB)
                        if img_array.ndim == 2:  # Grayscale image
                            img_array = np.stack([img_array] * 3, axis=-1)
                        elif img_array.shape[-1] == 4:  # RGBA
                            img_array = img_array[..., :3]  # Drop alpha channel

                        # Convert to (C, H, W)
                        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW

                        # Verify shape
                        expected_shape = (shape_meta['obs'][rgb_key]['shape'][0],) + target_size
                        if img_array.shape != expected_shape:
                            raise ValueError(f"Resized image shape {img_array.shape} doesn't match {expected_shape}")

                        all_images[rgb_key].append(img_array)
                    else:
                        raise FileNotFoundError(f"Image not found: {img_path}")

            # Create demo group for this episode
            demo_group = data_group.create_group(f'demo_{i}')
            obs_group = demo_group.create_group('obs')

            # Write lowdim data
            for key in lowdim_keys:
                lowdim_data = np.stack(all_data[key], axis=0)  # Shape: (T, ...)
                obs_group.create_dataset(key, data=lowdim_data, dtype=np.float32)

            # Write actions
            action_data = np.stack(all_data[action_key], axis=0)  # Shape: (T, ...)
            demo_group.create_dataset('actions', data=action_data, dtype=np.float32)

            # Write image data
            for key in rgb_keys:
                img_data = np.stack(all_images[key], axis=0)  # Shape: (T, C, H, W)
                obs_group.create_dataset(key, data=img_data, dtype=np.uint8)

    print(f"HDF5 file saved to {hdf5_path}")

if __name__ == "__main__":
    # Define shape metadata (updated for resized images: 240x320)
    shape_meta = {
        'obs': {
            'state': {'shape': (7,), 'type': 'low_dim'},
            'image_1': {'shape': (3, 240, 320), 'type': 'rgb'},  # (C, H, W)
            'image_2': {'shape': (3, 240, 320), 'type': 'rgb'},
        },
        'action': {'shape': (7,)}
    }

    # Define environment metadata
    env_meta = {
        'env_name': 'custom_robot_env',
        'type': 'robomimic',
        'env_kwargs': {}
    }

    # Paths
    input_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello"
    hdf5_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello_direct.h5"

    # Convert directly to HDF5 with resizing
    convert_custom_to_hdf5(input_path, hdf5_path, shape_meta, env_meta)