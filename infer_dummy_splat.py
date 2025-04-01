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
ckpt_path = '/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/outputs/2025.03.29/11.09.23_train_diffusion_unet_image_real_image/checkpoints/epoch=0350-train_loss=0.001.ckpt'
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
pred_action_list = [[] for _ in range(7)]  # 7 个维度的预测动作
gt_action_list = [[] for _ in range(7)]    # 7 个维度的真实动作
for start in range(100):
    image_0_1 = dataset.replay_buffer['image_1'][start]
    # image_0_1 = image_0_1.transpose(2, 0, 1)
    image_0_1 = np.expand_dims(image_0_1, axis=0)
    image_0_1 = np.expand_dims(image_0_1, axis=0)
    image_0_1 = torch.from_numpy(image_0_1).float()/255.0
    image_0_1 = image_0_1 + noise * torch.randn_like(image_0_1)

    image_0_3 = dataset.replay_buffer['image_2'][start]
    # image_0_3 = image_0_3.transpose(2, 0, 1)
    image_0_3 = np.expand_dims(image_0_3, axis=0)
    image_0_3 = np.expand_dims(image_0_3, axis=0)
    image_0_3 = torch.from_numpy(image_0_3).float()/255.0
    image_0_3 = image_0_3 + noise * torch.randn_like(image_0_3)

    image_1_1 = dataset.replay_buffer['image_1'][start + 1]
    # image_1_1 = image_1_1.transpose(2, 0, 1)
    image_1_1 = np.expand_dims(image_1_1, axis=0)
    image_1_1 = np.expand_dims(image_1_1, axis=0)
    image_1_1 = torch.from_numpy(image_1_1).float()/255.0
    image_1_1 = image_1_1 + noise * torch.randn_like(image_1_1)

    image_1_3 = dataset.replay_buffer['image_2'][start + 1]
    # image_1_3 = image_1_3.transpose(2, 0, 1)
    image_1_3 = np.expand_dims(image_1_3, axis=0)
    image_1_3 = np.expand_dims(image_1_3, axis=0)
    image_1_3 = torch.from_numpy(image_1_3).float()/255.0
    image_1_3 = image_1_3 + noise * torch.randn_like(image_1_3)


    last_state_obs = dataset.replay_buffer['action'][start][:].reshape(1, 1, 7)
    last_state_obs = torch.from_numpy(last_state_obs).float()

    cur_state_obs = dataset.replay_buffer['action'][start + 1][:].reshape(1, 1, 7)
    cur_state_obs = torch.from_numpy(cur_state_obs).float()

    # next_state_obs = dataset.replay_buffer['action'][start + 2][:].reshape(1, 1, 2)
    # next_state_obs = torch.from_numpy(next_state_obs).float()
    next_state_obs = dataset.replay_buffer['action'][start + 2]


    image_out_0 = torch.cat((image_0_1, image_1_1), dim=1)
    image_out_1 = torch.cat((image_0_3, image_1_3), dim=1)
    state_out = torch.cat((last_state_obs, cur_state_obs), dim=1)
        

    obs_dict_1 = {
        'image_1' :  image_out_0,
        'image_2' :  image_out_1,
        'joint_positions': state_out
    }
    result = policy.predict_action(obs_dict_1)
    action = result['action'][0].detach().to('cpu').numpy()
    # print('Action Pred:\n', action)
    # print('Action GT:\n', dataset.replay_buffer['action'][start+1:start+16][:])

    # pred action
    x1, x2,x3,x4,x5,x6,x7 = action[0]
    # gt action
    xt1, xt2,xt3,xt4,xt5,xt6,xt7 = dataset.replay_buffer['action'][start + 2][:]

    start_list.append(start)
    pred_action_list[0].append(x1)
    pred_action_list[1].append(x2)
    pred_action_list[2].append(x3)
    pred_action_list[3].append(x4)
    pred_action_list[4].append(x5)
    pred_action_list[5].append(x6)
    pred_action_list[6].append(x7)
    gt_action_list[0].append(xt1)
    gt_action_list[1].append(xt2)
    gt_action_list[2].append(xt3)
    gt_action_list[3].append(xt4)
    gt_action_list[4].append(xt5)
    gt_action_list[5].append(xt6)
    gt_action_list[6].append(xt7)

# 绘制对比图：7 个子图
plt.figure(figsize=(20, 14))  # 增大画布尺寸以容纳 7 个子图

# 定义维度标签
labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']

# 绘制 7 个子图（4 行 2 列布局，最后一列留空）
for i in range(7):
    plt.subplot(4, 2, i + 1)  # 4 行 2 列，索引从 1 开始
    plt.plot(start_list, pred_action_list[i], color='b', linestyle='-', label='Pred', alpha=0.8)
    plt.plot(start_list, gt_action_list[i], color='r', linestyle='--', label='GT', alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(f'Action Dimension {labels[i]}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 动态调整 y 轴范围
    all_values = pred_action_list[i] + gt_action_list[i]
    plt.ylim([min(all_values) - 0.1, max(all_values) + 0.1])

# 调整布局
plt.suptitle('Predicted vs Ground Truth Actions (All 7 Dimensions)', fontsize=16, y=1.02)  # 添加总标题
plt.tight_layout()
plt.savefig('dp_pusht_real_all_dims_subplots.png')