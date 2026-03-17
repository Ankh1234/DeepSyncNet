# DeepSyncNet: Deep Synchronized Fusion Network for EEG-fNIRS Multimodal BCI

[Paper (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1746809426005197) | [Dataset (Open Access EEG+fNIRS)](http://doc.ml.tu-berlin.de/hBCI)

<!-- Optional badges (may not render in some regions/networks) -->
[![Paper](https://img.shields.io/badge/Paper-BSPC%202026-blue)](https://www.sciencedirect.com/science/article/pii/S1746809426005197)
[![Dataset](https://img.shields.io/badge/Dataset-Open%20Access%20EEG%2BfNIRS-green)](http://doc.ml.tu-berlin.de/hBCI)

English | 中文

---


## 1. Overview | 项目简介

**English**

DeepSyncNet is an early-and-deep fusion framework for EEG-fNIRS hybrid BCI. The model integrates two modalities in low-level feature space and progressively refines fused representations via:

- **RFB** (Receptive Field Block)
- **AF** (Attentional Fusion / Gated Fusion)
- **FAM** (Feature Aggregation Module)
- **STA** (Spatiotemporal Attention)

Final prediction is adaptively combined using a learnable fusion weight.

**中文**

DeepSyncNet 是一个面向 EEG-fNIRS 混合脑机接口的“早期 + 深度”融合网络。模型在低层特征阶段融合两模态，并通过以下模块逐步增强特征：

- **RFB**（感受野块）
- **AF**（注意力融合 / 门控融合）
- **FAM**（特征聚合模块）
- **STA**（时空注意力模块）

最终通过可学习权重对分支输出进行自适应融合。

---

## 2. Paper & Dataset | 论文与数据集

- Paper (ScienceDirect): https://www.sciencedirect.com/science/article/pii/S1746809426005197
- Dataset: http://doc.ml.tu-berlin.de/hBCI
- GitHub: https://github.com/Ankh1234/DeepSyncNet

---

## 3. Core Pipeline | 核心流程

**Important / 重要说明**

Raw public data is provided in **`.mat`** format.
This repository uses MATLAB scripts first to convert `.mat` into Excel files, then Python scripts continue preprocessing/training/testing.

原始公开数据是 **`.mat`** 格式。
本项目先通过 MATLAB 脚本将 `.mat` 提取为 Excel，再由 Python 脚本完成后续预处理与训练测试。

### 3.1 End-to-end Flow | 全流程

1. **MATLAB extraction** (`.mat -> .xlsx`)  
   `Matlab_code/EEG.mlx`, `Matlab_code/fNIRS.mlx`
2. **Event time to sample index** (`label_time -> label_Hz`)  
   `BCI/EEG/label_time_to_Hz.py`, `BCI/fNIRS/label_time_to_Hz.py`
3. **EEG downsampling** (200 Hz -> 120 Hz)  
   `BCI/EEG/downsample.py`
4. **Spatial mapping to 16x16 + interpolation**  
   `data_extraction*.py`
5. **Train/Test tensor conversion (CSV -> .pt)**  
   `train_data_preprocessing.py`, `test_data_preprocessing.py`
6. **Multimodal training/testing**  
   `BCI/Multimodal/*.py`

### 3.2 MI/MA Session Split | MI/MA 任务会话划分

- **MI**: session `1,3,5`
- **MA**: session `2,4,6`

Both tasks share the same processing logic.
两种任务的数据处理方式一致，仅任务路径与会话索引不同。

---

## 4. Data Processing Details | 数据处理细节

### 4.1 MATLAB stage (`.mat -> Excel`)

`EEG.mlx` / `fNIRS.mlx` load `cnt.mat` and `mrk.mat`, and export:

- `label_time`: `mrk{1,b}.time`
- `label`: `mrk{1,b}.event.desc`
- `onehot_label`: `mrk{1,b}.y`
- `raw_data`: `cnt{1,b}.x`

### 4.2 Python stage (Excel -> model tensors)

- EEG is downsampled to **120 Hz**, fNIRS uses **10 Hz** timeline.
- Each time point is projected to a **16x16** spatial grid.
- Event-centered windows: **-2s to +7s**, total **10 windows**, each **3s**:
  - EEG window tensor: `(360, 16, 16)`
  - fNIRS window tensor: `(30, 16, 16)`
- Interpolation branch uses cubic + nearest completion.
- Final training/testing files are exported as `.pt` for PyTorch.

---

## 5. Environment | 环境依赖

### 5.1 Recommended

- Python 3.9+
- PyTorch (CUDA optional)
- MATLAB (for `.mlx` extraction stage)

### 5.2 Python packages

```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib seaborn tqdm openpyxl xlwt umap-learn ptflops
```

Optional for accelerated visualization:

```bash
pip install cuml
```

---

## 6. Path Configuration | 路径配置

Current scripts contain hard-coded absolute paths (e.g. `/root/autodl-tmp/project/...`).

当前脚本包含硬编码绝对路径（例如 `/root/autodl-tmp/project/...`）。

### Recommended migration strategy | 推荐迁移方式

1. Define your local project root and dataset root.
2. Replace hard-coded prefixes in scripts:
   - `/root/autodl-tmp/project` -> `<YOUR_PROJECT_ROOT>`
   - `/Users/lihao/Desktop/Open_Access_BCI_Data` -> `<YOUR_DATASET_ROOT>/Open_Access_BCI_Data`
3. Keep folder structure consistent with this repository.

You can still manually edit paths script by script if preferred.
你也可以按脚本手动替换路径。

---

## 7. Full Script Commands | 全脚本运行命令

Run from repository root:

```bash
cd /path/to/DeepSyncNet
```

### 7.1 MATLAB extraction (`.mat -> .xlsx`)

Open and run in MATLAB:

- `Matlab_code/EEG.mlx`
- `Matlab_code/fNIRS.mlx`

### 7.2 EEG preprocessing (standard pipeline)

```bash
python BCI/EEG/downsample.py
python BCI/EEG/label_time_to_Hz.py
python BCI/EEG/data_extraction.py
python BCI/EEG/data_extraction_testing.py
python BCI/EEG/train_data_preprocessing.py
python BCI/EEG/test_data_preprocessing.py
```

### 7.3 fNIRS preprocessing (standard pipeline)

```bash
python BCI/fNIRS/label_time_to_Hz.py
python BCI/fNIRS/data_extraction.py
python BCI/fNIRS/data_extraction_testing.py
python BCI/fNIRS/train_data_preprocessing.py
python BCI/fNIRS/test_data_preprocessing.py
```

### 7.4 DeepSyncNet training/testing

```bash
python BCI/Multimodal/Multimodal_train.py
python BCI/Multimodal/Multimodal_test.py
```

### 7.5 Ablation experiments

Module ablations:

```bash
python BCI/Multimodal/wo_RFB_train.py
python BCI/Multimodal/wo_RFB_test.py

python BCI/Multimodal/wo_GF_train.py
python BCI/Multimodal/wo_GF_test.py

python BCI/Multimodal/wo_FAM_train.py
python BCI/Multimodal/wo_FAM_test.py

python BCI/Multimodal/wo_STA_train.py
python BCI/Multimodal/wo_STA_test.py
```

No-interpolation ablation:

```bash
python BCI/EEG/EEG_no_interp.py
python BCI/EEG/EEG_no_interp_testing.py
python BCI/fNIRS/fNIRS_no_interp.py
python BCI/fNIRS/fNIRS_no_interp_testing.py
```

---

## 8. Main Results from Paper | 论文主要结果

### 8.1 Comparison with prior methods (Table 2)

| Algorithm | MA Mean Acc (%) | MA Max Acc (%) | MI Mean Acc (%) | MI Max Acc (%) |
|---|---:|---:|---:|---:|
| Rabbani et al. [48] | – | 82.76 | – | – |
| Zhe Sun et al. [40] | – | 90.19 | – | 77.53 |
| Xinyu Jiang et al. [27] | – | 91.15 | – | 78.56 |
| Yunyuan Gao et al. [49] | – | 92.24 | 79.48 | – |
| FGANet [39] | 91.96 ± 5.82 | 95.46 ± 5.12 | 78.59 ± 8.86 | 80.23 ± 9.63 |
| Qun He et al. [50] | – | – | 82.11 ± 7.25 | – |
| **DeepSyncNet (Ours)** | **98.40** | **98.53** | **98.92** | **99.21** |

### 8.2 Ablation study (Table 3)

#### MA task

| Model | Mean Acc (%) | Max Acc (%) |
|---|---:|---:|
| DeepSyncNet | 98.40 | 98.53 |
| without RFB | 97.17 | 97.50 |
| without AF | 90.48 | 90.60 |
| without FAM | 94.77 | 95.12 |
| without STA | 98.18 | 98.57 |
| without Interp. | 54.23 | 55.38 |

#### MI task

| Model | Mean Acc (%) | Max Acc (%) |
|---|---:|---:|
| DeepSyncNet | 98.92 | 99.21 |
| without RFB | 97.55 | 97.62 |
| without AF | 89.93 | 90.12 |
| without FAM | 95.25 | 95.48 |
| without STA | 97.61 | 97.86 |
| without Interp. | 56.56 | 58.33 |

---

## 9. Model Architecture | 模型结构

### 9.1 Baseline DeepSyncNet

- Main architecture: `BCI/Multimodal/Multimodal.py`
- Train/Test scripts:
  - `BCI/Multimodal/Multimodal_train.py`
  - `BCI/Multimodal/Multimodal_test.py`

Core modules in baseline:

- `RFB`: multi-branch receptive field extraction for EEG/fNIRS
- `AF` (implemented as `GatedFusion`): early attentional fusion
- `FAM`: feature aggregation/refinement
- `STA`: spatiotemporal attention refinement

Final fusion output in baseline:

```python
y_final = alpha * y_fused + (1 - alpha) * y_eeg
```

---

## 10. Ablation Design | 消融设计

#### A) `without RFB`

- Model script: `BCI/Multimodal/wo_RFB.py`
- Train/Test scripts:
  - `BCI/Multimodal/wo_RFB_train.py`
  - `BCI/Multimodal/wo_RFB_test.py`
- Ablation strategy:
  - Replace original RFB blocks with simple Conv3D + BN + ReLU feature extractor
  - Keep downstream modules (AF/FAM/STA/head) as consistent as possible

#### B) `without AF` (without Gated Fusion)

- Model script: `BCI/Multimodal/wo_GF.py`
- Train/Test scripts:
  - `BCI/Multimodal/wo_GF_train.py`
  - `BCI/Multimodal/wo_GF_test.py`
- Ablation strategy:
  - Remove gating/attention behavior in AF stage
  - Use simplified fusion path to preserve compatible tensor shapes for later blocks

#### C) `without FAM`

- Model script: `BCI/Multimodal/wo_FAM.py`
- Train/Test scripts:
  - `BCI/Multimodal/wo_FAM_train.py`
  - `BCI/Multimodal/wo_FAM_test.py`
- Ablation strategy:
  - Replace FAM with lightweight non-attentive substitute (`NoFAM`)
  - Keep tensor dimensions and following pipeline compatible

#### D) `without STA`

- Model script: `BCI/Multimodal/wo_STA.py`
- Train/Test scripts:
  - `BCI/Multimodal/wo_STA_train.py`
  - `BCI/Multimodal/wo_STA_test.py`
- Ablation strategy:
  - Replace STA with simplified non-attention module (`NoSTA`, mainly 1x1 Conv path)
  - Keep rest of architecture unchanged

#### E) `without Interpolation` (input-level ablation)

- EEG scripts:
  - `BCI/EEG/EEG_no_interp.py`
  - `BCI/EEG/EEG_no_interp_testing.py`
- fNIRS scripts:
  - `BCI/fNIRS/fNIRS_no_interp.py`
  - `BCI/fNIRS/fNIRS_no_interp_testing.py`
- Ablation strategy:
  - Remove spatial interpolation in preprocessing
  - Keep interpolation regions as zero to evaluate interpolation contribution

---

## 11. Citation | 引用

If this repository is useful for your research, please cite:

```bibtex
@article{LI2026109100,
  title = {DeepSyncNet: Deep synchronized fusion network for EEG-fNIRS multimodal brain-computer interfaces},
  journal = {Biomedical Signal Processing and Control},
  volume = {113},
  pages = {109100},
  year = {2026},
  issn = {1746-8094},
  doi = {https://doi.org/10.1016/j.bspc.2026.109100},
  url = {https://www.sciencedirect.com/science/article/pii/S1746809426005197},
  author = {Hao Li}
}
```
