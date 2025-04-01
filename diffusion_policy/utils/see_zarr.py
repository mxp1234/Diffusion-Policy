import zarr
import matplotlib.pyplot as plt
import numpy as np
import os

# 指定保存路径
output_dir = "/root/code/diffusion_policy/diffusion_policy/utils/imgs"
os.makedirs(output_dir, exist_ok=True)  

# 打开 Zarr 文件
zarr_path = "/root/code/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
group = zarr.open(zarr_path, 'r')
img_array = group['data']['img']

# 提取并归一化第一帧
frame_idx = 0
img = img_array[frame_idx]
img_min, img_max = img.min(), img.max()
if img_max > img_min:
    img = (img - img_min) / (img_max - img_min)

# 保存到指定路径
output_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
plt.imshow(img)
plt.title(f"Image at Frame {frame_idx}")
plt.axis('off')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"Saved image to {output_path}")