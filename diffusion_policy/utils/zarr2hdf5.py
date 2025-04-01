import zarr
import h5py
import numpy as np
from tqdm import tqdm
import json

def convert_zarr_to_hdf5(zarr_path, hdf5_path, env_meta):
    # Open Zarr file
    with zarr.ZipStore(zarr_path, mode='r') as zarr_store:
        zarr_root = zarr.group(zarr_store)
        zarr_data = zarr_root['data']
        zarr_meta = zarr_root['meta']

        # Get episode boundaries
        episode_ends = zarr_meta['episode_ends'][:]
        episode_starts = [0] + list(episode_ends[:-1])
        n_episodes = len(episode_ends)

        # Open HDF5 file for writing
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Create root groups
            data_group = hdf5_file.create_group('data')
            
            # Store env_meta as data.attrs['env_args'] (Robomimic expects this)
            data_group.attrs['env_args'] = json.dumps(env_meta)

            # Data keys
            lowdim_keys = ['joint_positions']
            rgb_keys = ['image_1', 'image_2']
            action_key = 'action'

            # Load all data into memory
            joint_positions = zarr_data['joint_positions'][:]
            actions = zarr_data['action'][:]
            image_1 = zarr_data['image_1'][:]
            image_2 = zarr_data['image_2'][:]

            # Write episode data
            for i in tqdm(range(n_episodes), desc="Converting episodes"):
                start = episode_starts[i]
                end = episode_ends[i]
                episode_len = end - start

                # Create demo group
                demo_group = data_group.create_group(f'demo_{i}')
                obs_group = demo_group.create_group('obs')

                # Write lowdim data
                obs_group.create_dataset('joint_positions', data=joint_positions[start:end], dtype=np.float32)
                demo_group.create_dataset('actions', data=actions[start:end], dtype=np.float32)

                # Write image data
                obs_group.create_dataset('image_1', data=image_1[start:end], dtype=np.uint8)
                obs_group.create_dataset('image_2', data=image_2[start:end], dtype=np.uint8)

    print(f"HDF5 file saved to {hdf5_path}")

# Example usage
if __name__ == "__main__":
    # zarr_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello.zarr.zip"
    zarr_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello_resize.zarr.zip"
    # hdf5_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello.h5"
    hdf5_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello_resize.h5"
    
    # Define environment metadata (adjust as needed)
    env_meta = {
        'env_name': 'custom_robot_env',  # Replace with your actual env name
        'type': 'robomimic',
        'env_kwargs': {}
    }
    
    convert_zarr_to_hdf5(zarr_path, hdf5_path, env_meta)