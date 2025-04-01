import torch
import dill
import hydra
import numpy as np
import cv2
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
# from diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy import DiffusionUnetImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


# load data
OmegaConf.register_new_resolver("eval", eval, replace=True)
with hydra.initialize('./diffusion_policy/config'):
    cfg = hydra.compose('train_diffusion_unet_real_image_workspace')
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)  # load h5

# load checkpoint
# ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.09.11/10.07.03_train_diffusion_unet_image_real_image_assembly/checkpoints/epoch=0550-train_loss=0.001.ckpt'
ckpt_path = '/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/outputs/2025.03.26/05.29.02_train_diffusion_unet_image_real_image/checkpoints/epoch=0350-train_loss=0.000.ckpt'
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# hacks for method-specific setup.
if 'diffusion' in cfg.name:
    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    policy.eval().to(device)

    # set inference params
    policy.num_inference_steps = 16 # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

noise = 0
start_list = []
pred_x_list = []
gt_x_list = []
pred_y_list = []
gt_y_list = []
for start in range(100):
    image_0_1 = dataset.replay_buffer['camera_1'][start]
    image_0_1 = image_0_1.transpose(2, 0, 1)
    image_0_1 = np.expand_dims(image_0_1, axis=0)
    image_0_1 = np.expand_dims(image_0_1, axis=0)
    image_0_1 = torch.from_numpy(image_0_1).float()/255.0
    image_0_1 = image_0_1 + noise * torch.randn_like(image_0_1)

    image_0_3 = dataset.replay_buffer['camera_3'][start]
    image_0_3 = image_0_3.transpose(2, 0, 1)
    image_0_3 = np.expand_dims(image_0_3, axis=0)
    image_0_3 = np.expand_dims(image_0_3, axis=0)
    image_0_3 = torch.from_numpy(image_0_3).float()/255.0
    image_0_3 = image_0_3 + noise * torch.randn_like(image_0_3)

    image_1_1 = dataset.replay_buffer['camera_1'][start + 1]
    image_1_1 = image_1_1.transpose(2, 0, 1)
    image_1_1 = np.expand_dims(image_1_1, axis=0)
    image_1_1 = np.expand_dims(image_1_1, axis=0)
    image_1_1 = torch.from_numpy(image_1_1).float()/255.0
    image_1_1 = image_1_1 + noise * torch.randn_like(image_1_1)

    image_1_3 = dataset.replay_buffer['camera_3'][start + 1]
    image_1_3 = image_1_3.transpose(2, 0, 1)
    image_1_3 = np.expand_dims(image_1_3, axis=0)
    image_1_3 = np.expand_dims(image_1_3, axis=0)
    image_1_3 = torch.from_numpy(image_1_3).float()/255.0
    image_1_3 = image_1_3 + noise * torch.randn_like(image_1_3)


    last_state_obs = dataset.replay_buffer['action'][start][:].reshape(1, 1, 2)
    last_state_obs = torch.from_numpy(last_state_obs).float()

    cur_state_obs = dataset.replay_buffer['action'][start + 1][:].reshape(1, 1, 2)
    cur_state_obs = torch.from_numpy(cur_state_obs).float()

    # next_state_obs = dataset.replay_buffer['action'][start + 2][:].reshape(1, 1, 2)
    # next_state_obs = torch.from_numpy(next_state_obs).float()
    next_state_obs = dataset.replay_buffer['action'][start + 2]


    image_out_0 = torch.cat((image_0_1, image_1_1), dim=1)
    image_out_1 = torch.cat((image_0_3, image_1_3), dim=1)
    state_out = torch.cat((last_state_obs, cur_state_obs), dim=1)
        

    obs_dict_1 = {
        'camera_1' :  image_out_0,
        'camera_3' :  image_out_1,
        'robot_eef_pose': state_out
    }
    result = policy.predict_action(obs_dict_1)
    action = result['action'][0].detach().to('cpu').numpy()
    # print('Action Pred:\n', action)
    # print('Action GT:\n', dataset.replay_buffer['action'][start+1:start+16][:])

    # pred action
    x, y = action[0]
    # gt action
    x2, y2 = dataset.replay_buffer['action'][start + 2][:]

    start_list.append(start)
    pred_x_list.append(x)
    gt_x_list.append(x2)
    pred_y_list.append(y)
    gt_y_list.append(y2)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(start_list, pred_x_list, label='action pred', marker='')
plt.plot(start_list, gt_x_list, label='action gt', marker='')
plt.ylim([0, 1])
plt.xlabel('step')
plt.ylabel('value')
plt.title('eef_pose1')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(start_list, pred_y_list, label='action pred', marker='')
plt.plot(start_list, gt_y_list, label='action gt', marker='')
plt.ylim([-0.5, 0.5])
plt.xlabel('step')
plt.ylabel('value')
plt.title('eef_pose2')
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('dp_pusht_real.png')
