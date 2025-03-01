import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import os

def process_fnirs_test_window(
    subject_id,
    window_id,
    data_csv_path,
    label_csv_path,
    data_pt_path,
    label_pt_path,
    n_samples=60,
    shape_3d=(30,16,16),
):

    # 1) 读取 CSV => (60,7680)
    df_data = pd.read_csv(data_csv_path, header=None)
    data_np = df_data.values
    print(f"[fNIRS test] sub0{subject_id}, window {window_id} => data shape:", data_np.shape)

    if data_np.shape[0] != n_samples:
        print(f"警告: 行数 {data_np.shape[0]} != n_samples={n_samples}")
    if data_np.shape[1] != np.prod(shape_3d):
        print(f"警告: 列数 {data_np.shape[1]} != shape_3d={shape_3d} 的乘积 {np.prod(shape_3d)}")

    # 3) reshape => (60,30,16,16)
    new_shape = (n_samples,) + shape_3d
    data_4d = data_np.reshape(new_shape)
    print("reshape =>", data_4d.shape)  # (60,30,16,16)

    # 4) 再加 channel=1 => (N,1,16,16,30)
    data_5d = data_4d[:, None, :, :, :]  # shape (N,1,16,16,30)
    print("final data shape =>", data_5d.shape)

    # 转 float tensor
    data_tensor = torch.from_numpy(data_5d).float()

    # 4) 读取标签 => (60,)
    df_label = pd.read_csv(label_csv_path, header=None)
    label_np = df_label.values
    print("label shape =>", label_np.shape)

    if label_np.shape[0] != n_samples:
        print(f"警告: 标签行数 {label_np.shape[0]} != n_samples={n_samples}")

    # 如果是(60,1) => flatten
    if len(label_np.shape)==2 and label_np.shape[1]==1:
        label_np = label_np[:,0]


    label_tensor = torch.from_numpy(label_np).long()  # =>(60,)

    print("final label shape =>", label_tensor.shape)

    # 5) 一次性保存
    torch.save(data_tensor, data_pt_path)
    torch.save(label_tensor, label_pt_path)
    print(f"[fNIRS test] sub0{subject_id}, window {window_id} => data saved: {data_pt_path}, label saved: {label_pt_path}")
    print("-----------------------------------------------------")


# =============== 用法示例 =================
if __name__=="__main__":
    for i in range(1,30):
        for l in range(10):
            data_csv  = f"/root/autodl-tmp/project/Open_Access_BCI_Data/fNIRS_data/testing_data/MA/sub0{i}_test_data_0{l}.csv"
            label_csv = f"/root/autodl-tmp/project/Open_Access_BCI_Data/fNIRS_data/testing_label/MA/sub0{i}_test_label_0{l}.csv"

            data_pt   = f"/root/autodl-tmp/project/BCI/fNIRS/pt_testing_4D_data/MA/sub0{i}_test4d_0{l}.pt"
            label_pt  = f"/root/autodl-tmp/project/BCI/fNIRS/test_target/MA/sub0{i}test_label_0{l}.pt"

            # 你在脚本中 reshape => (60,256,30) => =>(60,30,16,16)? 需要 check 256×30=7680 => 30×16×16=7680
            # => n_samples=60, shape_3d=(30,16,16)
            process_fnirs_test_window(
                subject_id   = i,
                window_id    = l,
                data_csv_path= data_csv,
                label_csv_path= label_csv,
                data_pt_path = data_pt,
                label_pt_path= label_pt,
                n_samples    = 60,             # CSV行数
                shape_3d     =(30,16,16),      # => (30×16×16=7680)
            )