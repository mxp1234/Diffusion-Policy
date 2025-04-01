import zarr
root = zarr.open("/root/code/diffusion_policy/data/real/bc_data_il/gello_all_apple.zarr", mode='r')
for key in ['image_1', 'image_2']:
    print(f"{key} shape: {root['data'][key].shape}")
print(root.tree())