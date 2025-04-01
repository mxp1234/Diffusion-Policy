import os
import glob
import pickle
import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image
import json
import concurrent.futures

def process_episode(demo_dir, i, shape_meta, target_size=(240, 320)):
    """处理单个 episode，返回数据供后续写入 HDF5"""
    pkl_files = sorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
    episode_len = len(pkl_files)

    if episode_len == 0:
        print(f"Warning: No .pkl files found in {demo_dir}, skipping.")
        return None

    # Extract metadata
    obs_shape_meta = shape_meta['obs']
    rgb_keys = [k for k, v in obs_shape_meta.items() if v.get('type', 'low_dim') == 'rgb']
    lowdim_keys = [k for k, v in obs_shape_meta.items() if v.get('type', 'low_dim') == 'low_dim']
    action_key = 'joint_positions'

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

        # Collect action from joint_positions
        if action_key in data:
            value = np.array(data[action_key], dtype=np.float32)
            expected_action_shape = shape_meta['action']['joint_positions_dummy']['shape']
            if value.shape != expected_action_shape:
                raise ValueError(f"Shape mismatch for action (joint_positions) in {pkl_file}: "
                               f"got {value.shape}, expected {expected_action_shape}")
            all_data[action_key].append(value)
        else:
            raise KeyError(f"Required key '{action_key}' not found in {pkl_file}")

        # Collect and resize images (改为 view_254 和 view_5)
        for cam_idx, rgb_key in enumerate(rgb_keys, 1):
            cam_dir = 'view_254' if cam_idx == 1 else 'view_5'
            img_path = os.path.join(demo_dir, cam_dir, f'{pkl_idx:05d}.png')
            if os.path.exists(img_path):
                img = Image.open(img_path).resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                img_array = np.array(img)  # Shape: (H, W, C)

                if img_array.ndim == 2:  # Grayscale image
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:  # RGBA
                    img_array = img_array[..., :3]  # Drop alpha channel

                img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
                expected_shape = (shape_meta['obs'][rgb_key]['shape'][0],) + target_size
                if img_array.shape != expected_shape:
                    raise ValueError(f"Resized image shape {img_array.shape} doesn't match {expected_shape}")

                all_images[rgb_key].append(img_array)
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

    # Prepare data for HDF5
    episode_data = {
        'lowdim': {key: np.stack(all_data[key], axis=0) for key in lowdim_keys},
        'actions': np.stack(all_data[action_key], axis=0),
        'images': {key: np.stack(all_images[key], axis=0) for key in rgb_keys},
        'demo_id': f'demo_{i}'
    }
    return episode_data

def convert_custom_to_hdf5(input_path, hdf5_path, shape_meta, env_meta, target_size=(240, 320), max_workers=None):
    # Default to number of CPU cores if max_workers not specified
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    # Get all subdirectories (episodes)
    demo_dirs = [d for d in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(d)]
    demo_dirs = sorted(demo_dirs)
    n_episodes = len(demo_dirs)

    if n_episodes == 0:
        raise ValueError(f"No episode directories found in {input_path}")

    print(f"Found {n_episodes} episode directories. Processing with {max_workers} threads.")

    # Process episodes in parallel
    episode_data_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_episode, demo_dir, i, shape_meta, target_size): i 
                         for i, demo_dir in enumerate(demo_dirs)}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=n_episodes, desc="Processing episodes"):
            episode_data = future.result()
            if episode_data is not None:
                episode_data_list.append(episode_data)

    # Write to HDF5 file
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        data_group = hdf5_file.create_group('data')
        data_group.attrs['env_args'] = json.dumps(env_meta)

        for episode_data in episode_data_list:
            demo_group = data_group.create_group(episode_data['demo_id'])
            obs_group = demo_group.create_group('obs')

            # Write lowdim data
            for key, lowdim_data in episode_data['lowdim'].items():
                obs_group.create_dataset(key, data=lowdim_data, dtype=np.float32)

            # Write actions
            demo_group.create_dataset('actions', data=episode_data['actions'], dtype=np.float32)

            # Write image data
            for key, img_data in episode_data['images'].items():
                obs_group.create_dataset(key, data=img_data, dtype=np.uint8)

    print(f"HDF5 file saved to {hdf5_path}")

if __name__ == "__main__":
    # Define shape metadata (updated for resized images: 240x320)
    shape_meta = {
        'obs': {
            'joint_positions': {'shape': (7,), 'type': 'low_dim'},
            'joint_positions_dummy': {'shape': (19,), 'type': 'low_dim'},
            'state': {'shape': (7,), 'type': 'low_dim'},
            'image_1': {'shape': (3, 240, 320), 'type': 'rgb'},  # (C, H, W)
            'image_2': {'shape': (3, 240, 320), 'type': 'rgb'},
        },
        'action': {
            'joint_positions': {'shape': (7,)}
        }
    }

    # Define environment metadata
    env_meta = {
        'env_name': 'custom_robot_env',
        'type': 'robomimic',
        'env_kwargs': {}
    }

    # Paths
    input_path = "/shared_disk/datasets/public_datasets/SplatSim/bc_data_view/gello"
    hdf5_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello_all.h5"

    # Convert with multithreading
    convert_custom_to_hdf5(input_path, hdf5_path, shape_meta, env_meta)
