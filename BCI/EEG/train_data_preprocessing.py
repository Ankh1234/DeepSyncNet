import pandas as pd
import numpy as np
import torch


def process_eeg_and_label_for_3dconv(
    data_csv_path,         # EEG数据CSV路径
    label_csv_path,        # 标签CSV路径
    data_pt_path,          # 数据.pt保存路径
    label_pt_path,         # 标签.pt保存路径
    n_samples,             # CSV中的样本数(行数)
    shape_4d=(360,16,16),  # 不含batch维度的三维(高,宽,深度/时间)
):

    # 1) 读取 CSV => DataFrame
    df_data = pd.read_csv(data_csv_path, header=None)
    print("原始数据 shape:", df_data.shape)  # (n_samples, shape_4d.prod())
    if df_data.shape[0] != n_samples:
        print(f"[警告] 数据行数 {df_data.shape[0]} != n_samples={n_samples}")
    if df_data.shape[1] != np.prod(shape_4d):
        print(f"[警告] 数据列数 {df_data.shape[1]} != shape_4d={shape_4d} 的乘积 {np.prod(shape_4d)}")

    # 转为 numpy array
    data_values = df_data.values

    # 3) reshape => (N, *shape_4d)  => (N,16,16,360)
    new_shape_4d = (n_samples,) + shape_4d
    data_4d = data_values.reshape(new_shape_4d)  # shape (N,16,16,360)

    # 4) 再加一个通道维 => (N,1,16,16,360)
    data_5d = data_4d[:, None, :, :, :]  # 在第二维插入 channel=1
    print("最终用于3D卷积的数据形状:", data_5d.shape)  # (N,1,16,16,360)

    # 转为 float tensor 并保存
    data_tensor = torch.from_numpy(data_5d).float()
    torch.save(data_tensor, data_pt_path)
    print(f"已保存 data => {data_pt_path}")

    # ============ 处理标签 =============
    df_label = pd.read_csv(label_csv_path, header=None)
    print("原始标签 shape:", df_label.shape)  # (n_samples,)

    label_values = df_label.values
    # flatten => (n_samples,)
    if len(label_values.shape)==2 and label_values.shape[1] ==1:
        label_values = label_values[:,0]

    if label_values.shape[0] != n_samples:
        print(f"[警告] 标签行数 {label_values.shape[0]} != n_samples={n_samples}")

    else:
        # 直接整数标签
        label_tensor = torch.from_numpy(label_values).long()  # (N,)

    torch.save(label_tensor, label_pt_path)
    print(f"已保存 label => {label_pt_path}")
    print("----------------------------------------------------")

# ================ 示例使用 ==================
if __name__ == "__main__":
    for i in range(1, 30):
        data_csv = f"/root/autodl-tmp/project/Open_Access_BCI_Data/EEG_data/extracted_data/MA/sub0{i}_extracted_data.csv"
        label_csv= f"/root/autodl-tmp/project/Open_Access_BCI_Data/EEG_data/extracted_label/MA/sub0{i}_extracted_label.csv"

        data_pt = f"/root/autodl-tmp/project/pt_training_4D_data/MA/sub0{i}_train4d.pt"
        label_pt= f"/root/autodl-tmp/project/train_target/MA/sub0{i}_train_target.pt"

        # 假设每个被试有 600 样本(行), shape_4d=(16,16,360)
        process_eeg_and_label_for_3dconv(
            data_csv_path = data_csv,
            label_csv_path= label_csv,
            data_pt_path  = data_pt,
            label_pt_path = label_pt,
            n_samples     = 600,       # 行数
            shape_4d      =(360,16,16),# H×W×D
        )