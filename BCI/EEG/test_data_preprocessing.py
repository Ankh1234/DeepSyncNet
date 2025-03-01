import pandas as pd
import numpy as np
import torch
import os


def process_test_window(
    subject_id,
    window_id,
    data_csv_path,
    label_csv_path,
    data_pt_path,
    label_pt_path,
):

    # 1) 读取数据 CSV => shape(60,92160)
    df_data = pd.read_csv(data_csv_path, header=None)
    data_np = df_data.values
    print(f"[sub0{subject_id}, window {window_id}] data shape =>", data_np.shape)

    # 检查行列
    if data_np.shape[0] != 60 or data_np.shape[1] != 92160:
        print(f"警告: data形状不符期望(60,92160) => {data_np.shape}")

    # 3) reshape => (60,16,16,360)
    data_4d = data_np.reshape((60,360,16,16))
    print(f"[sub0{subject_id}, window {window_id}] => 4D data shape:", data_4d.shape)

    # 转 float tensor
    data_tensor = torch.from_numpy(data_4d).float()

    # 4) 读取标签 => shape(60,)
    df_label = pd.read_csv(label_csv_path, header=None)
    label_np = df_label.values
    if label_np.shape[0] != 60:
        print(f"警告: label行数 != 60 => {label_np.shape}")
    # 若 shape(60,1) => flatten
    if len(label_np.shape)==2 and label_np.shape[1]==1:
        label_np = label_np[:,0]
    label_tensor = torch.from_numpy(label_np).long()

    print(f"[sub0{subject_id}, window {window_id}] => label shape:", label_tensor.shape)

    # 5) 保存
    torch.save(data_tensor, data_pt_path)
    torch.save(label_tensor, label_pt_path)
    print(f"已保存 => data:{data_pt_path} ; label:{label_pt_path}")
    print("-------------------------------------------------------")

# ============ 主函数示例 =============
if __name__=="__main__":
    for i in range(28,30):    # 被试 1..29
        for l in range(10): # 10 个时间窗
            # 拼路径
            data_csv  = f"/root/autodl-tmp/project/Open_Access_BCI_Data/EEG_data/testing_data/MA/sub0{i}_test_data_0{l}.csv"
            label_csv = f"/root/autodl-tmp/project/Open_Access_BCI_Data/EEG_data/testing_label/MA/sub0{i}_test_label_0{l}.csv"

            data_pt   = f"/root/autodl-tmp/project/pt_testing_4D_data/MA/sub0{i}_test4d_0{l}.pt"
            label_pt  = f"/root/autodl-tmp/project/test_target/MA/sub0{i}_test_label_0{l}.pt"

            process_test_window(
                subject_id    = i,
                window_id     = l,
                data_csv_path = data_csv,
                label_csv_path= label_csv,
                data_pt_path  = data_pt,
                label_pt_path = label_pt,
            )