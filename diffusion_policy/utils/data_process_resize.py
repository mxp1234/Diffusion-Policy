import os
import glob
import pickle
import numpy as np
import zarr
from tqdm import tqdm
from PIL import Image
import cv2  # 导入 OpenCV
from numcodecs import Blosc

try:
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
    register_codecs()
except ImportError:
    print("Warning: Jpeg2k codec not available, falling back to Blosc")
    Jpeg2k = None

def convert_custom_to_zarr(input_path, output_zarr_path, shape_meta):
    store = zarr.ZipStore(output_zarr_path, mode='w')
    root = zarr.group(store)
    data_group = root.require_group('data')
    meta_group = root.require_group('meta')

    obs_shape_meta = shape_meta['obs']
    rgb_keys = [k for k, v in obs_shape_meta.items() if v.get('type', 'low_dim') == 'rgb']
    lowdim_keys = [k for k, v in obs_shape_meta.items() if v.get('type', 'low_dim') == 'low_dim']

    demo_dirs = sorted(glob.glob(os.path.join(input_path, "[0-9]")))
    episode_ends = []
    episode_starts = [0]
    all_data = {key: [] for key in lowdim_keys + ['action']}
    all_images = {key: [] for key in rgb_keys}

    total_steps = 0
    # Process each demo
    for demo_dir in tqdm(demo_dirs, desc="Processing demo folders"):
        pkl_files = sorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
        episode_length = len(pkl_files)
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

            # Collect action
            if 'action' in data:
                value = np.array(data['action'], dtype=np.float32)
                if value.shape != shape_meta['action']['shape']:
                    raise ValueError(f"Shape mismatch for action in {pkl_file}: "
                                   f"got {value.shape}, expected {shape_meta['action']['shape']}")
                all_data['action'].append(value)
            else:
                raise KeyError(f"Required key 'action' not found in {pkl_file}")

            # Collect and resize images
            for cam_idx, rgb_key in enumerate(rgb_keys, 1):
                img_path = os.path.join(demo_dir, f'images_{cam_idx}', f'{pkl_idx:05d}.png')
                if os.path.exists(img_path):
                    # 使用 PIL 打开图像并转换为 NumPy 数组
                    img = np.array(Image.open(img_path))
                    # 使用 cv2 调整大小到 (320, 240)
                    img_resized = cv2.resize(img, (320, 240))  # (width, height)
                    # 注意：cv2.resize 返回的是 (height, width, channels)，但我们需要确保与 shape_meta 一致
                    all_images[rgb_key].append(img_resized)
                else:
                    raise FileNotFoundError(f"Image not found: {img_path}")

    n_steps = episode_ends[-1]
    if n_steps != total_steps:
        raise ValueError(f"Calculated steps {n_steps} doesn't match total steps {total_steps}")
    meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None)

    # Save lowdim data
    for key in tqdm(lowdim_keys + ['action'], desc="Saving lowdim data"):
        this_data = np.stack(all_data[key], axis=0)  # Explicitly stack along time axis
        print(f"Saving {key} with shape: {this_data.shape}")
        
        expected_shape = (n_steps,) + (shape_meta['action']['shape'] if key == 'action' 
                                     else shape_meta['obs'][key]['shape'])
        if this_data.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {key}: got {this_data.shape}, expected {expected_shape}")
        
        data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=(1, 7),
            compressor=None,
            dtype=np.float32
        )

    compressor = Jpeg2k(level=50) if Jpeg2k is not None else Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    for key in tqdm(rgb_keys, desc="Saving image data"):
        # 更新为调整后的目标形状 (n_steps, 240, 320, c)
        c = shape_meta['obs'][key]['shape'][0]  # 通道数保持不变
        img_data = np.stack(all_images[key], axis=0)
        print(f"Saving {key} with shape: {img_data.shape}")
        if img_data.shape[0] != n_steps:
            raise ValueError(f"Number of images ({img_data.shape[0]}) doesn't match steps ({n_steps})")
        
        img_arr = data_group.create_dataset(
            name=key,
            shape=(n_steps, 240, 320, c),  # 更新为调整后的尺寸
            chunks=(1, 240, 320, c),
            compressor=compressor,
            dtype=np.uint8
        )
        
        for i in range(n_steps):
            img_arr[i] = all_images[key][i]

    store.close()
    with zarr.ZipStore(output_zarr_path, mode='r') as verify_store:
        root = zarr.group(verify_store)
        for key in ['state', 'action', 'image_1', 'image_2']:
            arr = root['data'][key]
            print(f"Loaded {key} shape: {arr.shape}")
            if key in ['state', 'action']:
                print(f"Loaded {key} first 2 timesteps: {arr[:2]}")
    print(f"Zarr file saved to {output_zarr_path}")

if __name__ == "__main__":
    shape_meta = {
        'obs': {
            'state': {'shape': (7,), 'type': 'low_dim'},
            'image_1': {'shape': (3, 240, 320), 'type': 'rgb'},  # 更新为目标尺寸
            'image_2': {'shape': (3, 240, 320), 'type': 'rgb'},  # 更新为目标尺寸
        },
        'action': {'shape': (7,)}
    }
    
    input_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello"
    output_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello_state_resize.zarr"
    
    convert_custom_to_zarr(input_path, output_path, shape_meta)