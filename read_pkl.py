# results/test_protocol_1_0504/test_videos中的0、274、313、371可作攻击对象


import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt

# pkl_file_path = './data/real_captured_data/0000.pkl'
# pkl_file_path = './data/protocol_1/CMU/test/1.pkl'
pkl_file_path = './adversarial_example/ae1_noattack.pkl'

# 读取.pkl文件
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# 'hmd_position_global_full_gt_list'是输出label，有3*132*1570个值
# data['body_parms_list']['root_orient']有3*1570个值
# data['body_parms_list']['pose_body']有63*1570个值
# data['body_parms_list']['trans']有3*1570个值
# 'head_global_trans_list'是1570个刚体变换矩阵
# 'rotation_local_full_gt_list'是输出label，有132*1570个值
# 132是输出维度
# 1570是视频帧数
# data['hmd_position_global_full_gt_list']即为input signal

seq_len = data['hmd_position_global_full_gt_list'].shape[0]
# print('seq_len:', seq_len)

# data_dict =  {'input_signal': data['hmd_position_global_full_gt_list'].reshape(seq_len, -1).float(),
#         'rotation_local_full': data['rotation_local_full_gt_list'], 
#         'body_param_list': data['body_parms_list'],
#         'global_head_trans': data['head_global_trans_list'],
#         'pos_pelvis_gt': data['body_parms_list']['trans'],
#         'floor_height': 0, 
#         }
# print('打印第0帧Input Signal:')
# print(data_dict['input_signal'][0])
# print(len(data_dict['input_signal'][0]))


# input_signal = data['input_signal']
data_orig = data['hmd_position_global_full_gt_list']
# print('data_orig.shape:', data_orig.shape)
# print('data_orig:', data_orig)

input_signal = data['hmd_position_global_full_gt_list'].reshape(seq_len, -1).float()

# print('input signal.shape:', input_signal.shape)
# print('input signal:', input_signal)


input_signal = input_signal.unsqueeze(0)
print('input signal.shape:', input_signal.shape)

batch, seq_len = input_signal.shape[0], input_signal.shape[1]
print('batch:', batch)
print('seq_len:', seq_len)

rotation = input_signal[:, :, :22*6].reshape(batch, seq_len, 22, 6) # 22代表22个关节，其中只有三个关节有数据，分别代表头和两只手，6代表6个维度
velocity_rotation = input_signal[:, :, 22*6:22*6*2].reshape(batch, seq_len, 22, 6)
position = input_signal[:, :, 22*6*2:22*6*2+3*22].reshape(batch, seq_len, 22, 3)
velocity_position = input_signal[:, :, 22*6*2+3*22:].reshape(batch, seq_len, 22, 3)

# print('rotation:', rotation[0][1])
print('rotation:', rotation[0][1][15]) # 1代表第1帧，15代表第15个关节(头)，6个维度
print('rotation:', rotation[0][1][20]) # 1代表第1帧，20代表第20个关节(左手)，6个维度
print('rotation:', rotation[0][1][21]) # 1代表第1帧，21代表第21个关节(右手)，6个维度

# print('velocity_rotation:', velocity_rotation[0][1][15]) # 1代表第1帧，15代表第15个关节(头)，6个维度




# --------------------------------------------------------
# 画图

rotation_right_hand = rotation[0, :, 21]  # Select rotation for right hand

# print('rotation_right_hand:', rotation_right_hand)

# Reshape rotation_right_hand to have shape (batch * seq_len, 6)
rotation_right_hand = rotation_right_hand.reshape(-1, 6)
# print('rotation_right_hand:', rotation_right_hand)

# Create time steps for x-axis
fps = 60.0
time_steps = np.linspace(0,seq_len/fps, seq_len)

# Plot rotation for each dimension
plt.figure(0)
for i in range(3,6):
# for i in range(3):
    plt.plot(time_steps, rotation_right_hand[:, i], label=f'Dimension {i+1+3}')

plt.xlabel('Time')
plt.ylabel('Rotation')
plt.title('Rotation of Right Hand')
plt.legend()
# plt.show()
plt.savefig('./rotation_right_hand.png')

# --------------------------------------------------------




# --------------------------------------------------------
# 给Dimension 3注入噪声
ad_input_signal = copy.deepcopy(data_orig)

# print('ad_input_signal.shape:', ad_input_signal.shape)

ad_rotation = ad_input_signal[ :, :22*6].reshape(seq_len, 22, 6) # 22代表22个关节，其中只有三个关节有数据，分别代表头和两只手，6代表6个维度


# 修改Dimension 3
time_steps = np.linspace(0, 60, 60)
amplitude = 0.65
frequency = 1/30

sin_signal = amplitude * np.sin(2*np.pi*frequency * time_steps)
# sin_signal = sin_signal.reshape(60, 1,1)
print('sin_signal.shape:', sin_signal.shape)

# ad_rotation[132:192][21][3] = ad_rotation[132:192][21][3] + sin_signal
for i in range(132, 192):
    ad_rotation[i][21][4] = ad_rotation[i][21][4] + sin_signal[i-132]






# 绘制ad_rotation[:][21][3]

# plt.plot(ad_rotation[:][21][3])

rotation_right_hand = ad_rotation[:, 21]  # Select rotation for right hand


# Reshape rotation_right_hand to have shape (batch * seq_len, 6)
rotation_right_hand = rotation_right_hand.reshape(-1, 6)

# Create time steps for x-axis
fps = 60.0
time_steps = np.linspace(0,seq_len/fps, seq_len)

# Plot rotation for each dimension
plt.figure(1)
for i in range(3,6):
# for i in range(3):
    plt.plot(time_steps, rotation_right_hand[:, i], label=f'Dimension {i+1+3}')

plt.xlabel('Time')
plt.ylabel('Rotation')
plt.title('[AE] Rotation of Right Hand')
plt.legend()
# plt.show()
plt.savefig('./ad_rotation_right_hand.png')







ad_input_signal[ :, :22*6] = ad_rotation.reshape(seq_len, 22*6)


data['hmd_position_global_full_gt_list'] = ad_input_signal

pkl_file_path = pkl_file_path.replace('ae1_noattack', 'ae1_attack')
print('attack_pkl_file_path:', pkl_file_path)


# 将data写回pkl文件
with open(pkl_file_path, 'wb') as f:
    pickle.dump(data, f)

# # 读取.pkl文件
# with open(pkl_file_path, 'rb') as f:
#     data = pickle.load(f)

# print(data['hmd_position_global_full_gt_list'][100])