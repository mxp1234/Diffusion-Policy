from typing import Dict, List
import torch
import numpy as np
from tqdm import tqdm
import zarr
import os
import shutil
import copy
from filelock import FileLock
from threadpoolctl import threadpool_limits
from numcodecs import Blosc
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

register_codecs()

class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + '.cache.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    print('Cache does not exist. Creating!')
                    replay_buffer = self._load_zarr_to_replay(
                        store=zarr.MemoryStore(),
                        dataset_path=dataset_path,
                        shape_meta=shape_meta,
                        abs_action=abs_action,
                        rotation_transformer=rotation_transformer
                    )
                    print('Saving cache to disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='w') as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = self._load_zarr_to_replay(
                store=zarr.MemoryStore(),
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer
            )

        rgb_keys = ['image_1', 'image_2']
        lowdim_keys = ['joint_positions']

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def _load_zarr_to_replay(self, store, dataset_path, shape_meta, abs_action, rotation_transformer):
        with zarr.ZipStore(dataset_path, mode='r') as src_store:
            src_root = zarr.group(src_store)
            src_data = src_root['data']
            src_meta = src_root['meta']

            root = zarr.group(store)
            data_group = root.require_group('data', overwrite=True)
            meta_group = root.require_group('meta', overwrite=True)

            episode_ends = src_meta['episode_ends'][:]
            n_steps = episode_ends[-1]
            meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None, overwrite=True)

            rgb_keys = ['image_1', 'image_2']
            lowdim_keys = ['joint_positions']

            for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
                this_data = src_data[key][:]
                if key == 'action' and abs_action:
                    this_data = _convert_actions(
                        raw_actions=this_data,
                        abs_action=abs_action,
                        rotation_transformer=rotation_transformer
                    )
                # Convert shape_meta shapes to tuples for comparison
                shape_key = 'action' if key == 'action' else key
                expected_inner_shape = tuple(shape_meta['action']['shape'] if key == 'action' 
                                           else shape_meta['obs'][key]['shape'])
                expected_shape = (n_steps,) + expected_inner_shape
                if this_data.shape != expected_shape:
                    raise ValueError(f"Shape mismatch for {key}: got {this_data.shape}, expected {expected_shape}")
                data_group.array(
                    name=key,
                    data=this_data,
                    shape=this_data.shape,
                    chunks=(1, this_data.shape[1]),
                    compressor=None,
                    dtype=np.float32
                )

            compressor = Jpeg2k(level=50) if Jpeg2k is not None else Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
            for key in tqdm(rgb_keys, desc="Loading image data"):
                c, h, w = tuple(shape_meta['obs'][key]['shape'])  # Convert to tuple
                img_data = src_data[key][:]
                expected_shape = (n_steps, h, w, c)
                if img_data.shape != expected_shape:
                    raise ValueError(f"Shape mismatch for {key}: got {img_data.shape}, expected {expected_shape}")
                data_group.create_dataset(
                    name=key,
                    data=img_data,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=compressor,
                    dtype=np.uint8
                )

        replay_buffer = ReplayBuffer(root)
        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('positions'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f'Unsupported lowdim key: {key}')
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)
        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = raw_actions.shape[-1] == 14
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 2, 7)
        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
        if is_dual_arm:
            actions = actions.reshape(-1, 20)
    return actions

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

if __name__ == "__main__":
    shape_meta = {
        'obs': {
            'joint_positions': {'shape': (7,), 'type': 'low_dim'},
            'image_1': {'shape': (3, 1071, 1907), 'type': 'rgb'},
            'image_2': {'shape': (3, 1071, 1907), 'type': 'rgb'},
        },
        'action': {'shape': (7,)}
    }
    dataset_path = "/root/code/diffusion_policy/data/real/bc_data_il/gello.zarr.zip"
    dataset = RobomimicReplayImageDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=16,
        n_obs_steps=2
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")