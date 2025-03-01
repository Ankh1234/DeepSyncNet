from sklearn.model_selection import KFold
import numpy as np
from Multimodal import *

# 数据
data = np.arange(1,30)

kf = KFold(n_splits=5, shuffle=True, random_state=Config.seed)


folds = [{"train": train_idx, "test": test_idx} for train_idx, test_idx in kf.split(data)]



