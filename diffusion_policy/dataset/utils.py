import zarr
root = zarr.open("/root/code/diffusion_policy/data/real/bc_data_il/gello_all.zarr", mode='r')
print(root['data']['image_1'].shape)