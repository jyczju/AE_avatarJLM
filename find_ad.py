import os
import pickle
import shutil

def traverse_folder(folder_path):
    pkl_file_path_list = []
    for root, dirs, files in os.walk(folder_path):
        if 'train' in root:
            continue
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                # print(file_path)
                pkl_file_path_list.append(file_path)
    return pkl_file_path_list

def get_seq_len(pkl_file_path):
    # pkl_file_path = './adversarial_example/1.pkl'

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
    return seq_len


if __name__ == '__main__':
    folder_path = "./data/protocol_1"
    pkl_file_path_list = traverse_folder(folder_path)
    for pkl_file_path in pkl_file_path_list:
        seq_len = get_seq_len(pkl_file_path)
        if 420 <= seq_len <= 430:
            print(pkl_file_path)
            print('seq_len:', seq_len)
            # 将文件复制到指定文件夹
            destination_folder = "./adversarial_example"
            shutil.copy(pkl_file_path, destination_folder)





