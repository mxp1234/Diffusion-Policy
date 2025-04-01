import zipfile
import pickle
import os
'''观察一个pkl'''

extract_dir = "/mnt/data-3/users/mengxinpan/"


# 读取 .pkl 文件
pkl_path = os.path.join(extract_dir, "code/diffusion_policy/data/real/bc_data_il/gello/3/00001.pkl")
if not os.path.exists(pkl_path):
    print(f"Error: {pkl_path} not found!")
else:
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        print("Loaded data:", data)
    

    
    
# {
#     'joint_positions': [ 0.10400249, -1.27832248,  1.39666304, -1.57140399, -1.56695457, 0.23855038,  0.        ], 
#     'all_joint_positions': [ 0.10400249, -1.27832248,  1.39666304, -1.57140399, -1.56695457,
#         0.23855038,  0.        ,  0.        ,  0.24190453,  0.        ,
#        -0.29585218,  0.        ,  0.25283451,  0.21198826,  0.        ,
#        -0.27606276,  0.        ,  0.25462059,  0.        ], 
#     'joint_velocities': [ 0.        , -0.30000305, -0.29995271,  0.3002189 ,  0.00119601,
#         0.00045537, -0.29677954], 
#     'joint_positions_dummy': [ 0.10400249, -1.27832248,  1.39666304, -1.57140399, -1.56695457,
#         0.23855038,  0.        ,  0.        ,  0.24190453,  0.        ,
#        -0.29585218,  0.        ,  0.25283451,  0.21198826,  0.        ,
#        -0.27606276,  0.        ,  0.25462059,  0.        ], 
#     'state': [ 0.5917231 ,  0.17226873,  0.43852979, -1.68608534, -0.02419326,
#        -1.70416086,  0.30238066], 
#     'action': [ 0.5917231 ,  0.17226873,  0.43852979, -1.68608534, -0.02419326,
#        -1.70416086,  1.        ], 
#     'plastic_apple_position': (0.5663283892554869, 0.17965189060530434, 0.1608330973265537), 
#     'plastic_apple_orientation': (0.028589646599495658, 0.036828054861807104, -0.009407734125730914, 0.9988682701042372)
#     }

# {'joint_positions': [ 0.07275212, -1.30956756,  1.42793584, -1.57127755, -1.56690938,0.2076222 ,  0.        ], 
#  'all_joint_positions': [ 0.07275212, -1.30956756,  1.42793584, -1.57127755, -1.56690938, 0.2076222 ,  0.        ,  0.        ,  0.24104262,  0.        ,
#        -0.29461735,  0.        ,  0.25187919,  0.21148536,  0.        ,-0.27498826,  0.        ,  0.25367494,  0.        ]
#  'joint_velocities': [ 0.        , -0.30000406, -0.29995275,  0.30021889,  0.00122965,0.00041359, -0.29703201], 
#  'joint_positions_dummy': [ 0.07275212, -1.30956756,  1.42793584, -1.57127755, -1.56690938,0.2076222 ,  0.        ,  0.        ,  0.24104262,  0.        ,
#        -0.29461735,  0.        ,  0.25187919,  0.21148536,  0.        ,
#        -0.27498826,  0.        ,  0.25367494,  0.        ],
# 'state': [ 0.5840803 ,  0.15275182,  0.44214989, -1.68693397, -0.02061245,-1.70468496,  0.30130327],
# 'action': [ 0.5840803 ,  0.15275182,  0.44214989, -1.68693397, -0.02061245,-1.70468496,  1.        ], 
# 'plastic_apple_position': (0.5585600154401069, 0.16113685375886197, 0.1644903727603417), 
# 'plastic_apple_orientation': (0.030190237588723814, 0.0369847711339731, -0.00958862243406686, 0.9988136635921242)}