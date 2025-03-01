# 处理数据
# import pandas as pd
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import os

def process_fnirs_3dconv(
    subject_id,
    data_csv_path,
    label_csv_path,
    data_pt_path,
    label_pt_path,
    n_samples,              # CSV中的样本行数, e.g. 600
    shape_3d=(30,16,16),   # 不含batch维度的三维，(depth=16, height=16, width=360) 只是举例
):

    # 1) 读取数据 CSV => shape (n_samples, 16*16*360=92160)
    df_data = pd.read_csv(data_csv_path, header=None)
    data_np = df_data.values
    print(f"[fNIRS] sub0{subject_id} => data shape:", data_np.shape)

    if data_np.shape[0] != n_samples:
        print(f"警告: 行数 {data_np.shape[0]} != n_samples={n_samples}")
    if data_np.shape[1] != np.prod(shape_3d):
        print(f"警告: 列数 {data_np.shape[1]} != shape_3d={shape_3d} 的乘积 {np.prod(shape_3d)}")

    # 3) reshape => (n_samples, 16,16,360)
    #    这里16×16×360=92160
    new_shape_4d = (n_samples,) + shape_3d  # (N,16,16,30)
    data_4d = data_np.reshape(new_shape_4d)
    print("reshape =>", data_4d.shape)  # (N,16,16,30)

    # 4) 再加 channel=1 => (N,1,16,16,30)
    data_5d = data_4d[:,None,:,:,:]  # shape (N,1,16,16,30)
    print("final data shape =>", data_5d.shape)

    # 转 float tensor
    data_tensor = torch.from_numpy(data_5d).float()

    # 5) 读取标签 => shape (n_samples,)
    df_label = pd.read_csv(label_csv_path, header=None)
    label_np = df_label.values
    print("label shape =>", label_np.shape)
    if label_np.shape[0] != n_samples:
        print(f"警告: 标签行数 {label_np.shape[0]} != n_samples={n_samples}")

    # 若 (n_samples,1) => flatten
    if len(label_np.shape)==2 and label_np.shape[1]==1:
        label_np = label_np[:,0]

    label_tensor = torch.from_numpy(label_np).long()  # (N,)

    print("final label shape =>", label_tensor.shape)

    # 6) 只做一次保存
    torch.save(data_tensor, data_pt_path)
    torch.save(label_tensor, label_pt_path)
    print(f"[fNIRS] sub0{subject_id} data => {data_pt_path}, label => {label_pt_path}")
    print("---------------------------------------------------")

# ------------------------------
# 示例: 在 main() 中对 i in [1..29] 进行
if __name__=="__main__":
    for i in range(1,30):
        data_csv  = f"/root/autodl-tmp/project/Open_Access_BCI_Data/fNIRS_data/extracted_data/MA/sub0{i}_extracted_data.csv"
        label_csv = f"/root/autodl-tmp/project/Open_Access_BCI_Data/fNIRS_data/extracted_label/MA/sub0{i}_extracted_label.csv"

        data_pt   = f"/root/autodl-tmp/project/BCI/fNIRS/pt_training_4D_data/MA/sub0{i}_train4d.pt"
        label_pt  = f"/root/autodl-tmp/project/BCI/fNIRS/train_target/MA/sub0{i}train_target.pt"

        # 假设 n_samples=600, shape_3d=(16,16,360)=92160
        process_fnirs_3dconv(
            subject_id   = i,
            data_csv_path= data_csv,
            label_csv_path= label_csv,
            data_pt_path = data_pt,
            label_pt_path= label_pt,
            n_samples    = 600,
            shape_3d     = (30,16,16),
        )