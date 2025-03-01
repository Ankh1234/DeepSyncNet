from sklearn.model_selection import KFold
import numpy as np

# 数据
data = np.arange(1, 30)

# 2 折交叉验证
kf = KFold(n_splits=2, shuffle=True, random_state=42)

# 生成 2 折的训练集和测试集索引
folds = [{"train": train_idx, "test": test_idx} for train_idx, test_idx in kf.split(data)]

