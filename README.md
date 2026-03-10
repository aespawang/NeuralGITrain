# Train

## Conda 环境（CUDA 11.8 GPU 版 PyTorch）

在本目录下执行：

```bash
conda env create -f environment.yml -y
conda activate neuralgi-train-cu118
```

如果你修改过 `environment.yml`，想更新现有环境：

```bash
conda env update -f environment.yml --prune -y
```

1. run `apv_train_dataset_maker.py`
2. run `apv_train.py`
3. run `apv_eval.py`
4. run `exr_compare.py`