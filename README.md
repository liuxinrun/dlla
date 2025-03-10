
## DLLA: Dynamic Learnable Label Assignment for Indoor 3D Object Detection

This repository contains an implementation of DLLA, a dynamiclearnable label assignment for indoor 3D Object Detection method introduced in our paper:

> **Dynamic Learnable Label Assignment for Indoor 3D Object Detection**<br>
> [Xinrun Liu](https://github.com/liuxinrun),
> [Linqin Zhao](https://github.com/lqzhao),
> [Bin Fan](https://scholar.google.com/citations?hl=zh-CN&user=8z7e7aYAAAAJ),
> [Jiwen Lu](https://scholar.google.com/citations?hl=zh-CN&user=TN8uDQoAAAAJ),
> [Hongmin Liu](https://scholar.google.com/citations?hl=zh-CN&user=iMcqErcAAAAJ)
> <br>
> University of Science and Technology Beijing, Tsinghua University <br>

### Installation

You can install all required packages manually. This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework.
Please refer to the original installation guide [getting_started.md](docs/getting_started.md), including MinkowskiEngine installation, replacing `open-mmlab/mmdetection3d` with `liuxinrun/dlla`.


Most of the `DLLA`-related code locates in the following files: 
[detectors/mink_single_stage.py](mmdet3d/models/detectors/mink_single_stage.py),
[dense_heads/dlla_tr3d_head.py](mmdet3d/models/dense_heads/dlla_tr3d_head.py),
[dense_heads/dlla_fcaf3d_head.py](mmdet3d/models/dense_heads/dlla_fcaf3d_head.py),

### Getting Started

Please see [getting_started.md](docs/getting_started.md) for basic usage examples.
We follow the mmdetection3d data preparation protocol described in [scannet](data/scannet), [sunrgbd](data/sunrgbd).

**Training**

To start training, run [train](tools/train.py) with DLLA for TR3D [configs](configs/dlla):
```shell
python tools/train.py configs/dlla/dlla_tr3d_scannet-3d-18class.py
```

Run [train](tools/train.py) with DLLA for FCAF3D [configs](configs/dlla):
```shell
python tools/train.py configs/dlla/dlla_fcaf3d_scannet-3d-18class.py
```

**Testing**

Test pre-trained model using [test](tools/dist_test.sh) with DLLA for TR3D [configs](configs/dlla):
```shell
python tools/test.py configs/dlla/dlla_tr3d_scannet-3d-18class.py \
    work_dirs/dlla_tr3d_scannet-3d-18class/latest.pth --eval mAP
```

**Visualization**

Visualizations can be created with [test](tools/test.py) script. 
For better visualizations, you may set `score_thr` in configs to `0.3`:
```shell
python tools/test.py configs/dlla/dlla_tr3d_scannet-3d-18class.py \
    work_dirs/dlla_tr3d_scannet-3d-18class/latest.pth --eval mAP --show \
    --show-dir work_dirs/dlla_tr3d_scannet-3d-18class
```

### Models

The metrics are obtained in 15 training runs followed by 15 test runs. We report both the best and the average values (the latter are given in round brackets).

**DLLA 3D Detection**

| Model | mAP@0.25 | mAP@0.5 |
|:-------:|:--------:|:-------:|
| DLLA_TR3D | 73.8 (72.8) | 60.2 (58.9) |
| DLLA_FCAF3D | 71.4 (71.0) | 60.0 (59.0) |

The corresponding model, config, and log can be found at the link below:
https://yunpan.ustb.edu.cn/link/AA237DB1AB6BE54E989296409225ECA2C9
