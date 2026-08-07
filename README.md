# MonoFusion / MonoFusion-R

### Digging into Monocular Texture-Scale Feature Fusion for Optical Flow and Optical Expansion Estimation

**Changlin Li, Yeqiang Qian, Chunxiang Wang, Ming Yang**
Department of Automation, Shanghai Jiao Tong University

**IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2026**

Official PyTorch implementation of **MonoFusion** and its recurrent variant **MonoFusion-R**: a texture-scale feature fusion pipeline that estimates monocular **optical flow (OF)** and **optical expansion (OE)** as two equally-important, mutually-reinforcing tasks, instead of treating OE as a by-product of OF.

<p align="center">
  <img src="./images/pipeline.png" width="850" alt="MonoFusion pipeline"/>
</p>

---

## Table of Contents

- [Highlights](#highlights)
- [MonoFusion vs. MonoFusion-R](#monofusion-vs-monofusion-r)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Datasets](#datasets)
- [Model Zoo](#model-zoo)
- [Training](#training)
- [Evaluation & Inference](#evaluation--inference)
- [Results](#results)
- [Visualization](#visualization)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Highlights

Existing dense optical-expansion methods derive expansion from optical flow, or estimate the two tasks from a largely shared representation, so OE accuracy is bottlenecked by OF quality. We instead treat **texture** (color/shape, driving OF) and **scale** (projected-size change, driving OE) as two independent, same-level feature streams that guide each other:

- **MonoFusion**: independently extracts texture and scale features and fuses/updates them with a dedicated fusion encoder-decoder, so optical flow and optical expansion mutually enhance each other instead of one being derived from the other.
- **MonoFusion-R**: wraps the same fusion encoder/decoder in a lightweight, weight-shared **recurrent loop** that progressively refines texture and scale features together, trading a few extra iterations for a sizeable accuracy gain (see [below](#monofusion-vs-monofusion-r)).
- State-of-the-art **monocular** results on four KITTI tasks — optical flow, optical expansion, time-to-collision (TTC), and scene flow — and strong optical-flow generalization on FlyingThings3D.

## MonoFusion vs. MonoFusion-R

MonoFusion-R reuses **every module of MonoFusion unchanged** (same fusion encoder, same fusion decoder, same weights). The only difference is *how many times* the fusion decoder is applied: each recurrent iteration (i) updates the scale-conditioned texture features via optical-flow warping, (ii) re-fuses texture and scale and re-estimates flow/expansion through the fusion decoder, and (iii) feeds the new estimate back into the next iteration — all iterations sharing the same weights (RNN-style), so no extra parameters are introduced.

<p align="center">
  <img src="./images/monofusion_r.png" width="750" alt="MonoFusion-R recurrent framework"/>
</p>

<p align="center"><i>Fig. MonoFusion-R: the Fusion Encoder feeds a recurrent loop of texture updating &rarr; feature fusion + OF/OE estimation &rarr; scale updating.</i></p>

Because the recurrent loop is weight-shared, the number of iterations is a pure **inference-time knob** exposed as `--num_iterations` (`MonoFusion.num_iterations` / `ScaleNet.num_iterations` in code) — no architecture change or extra parameters are required to move between MonoFusion (`--num_iterations 1`) and MonoFusion-R (`--num_iterations > 1`). Accuracy keeps improving with more iterations, at a modest speed cost, and already surpasses the strongest recurrent baseline (TPCV, 12 GRU iterations) after only 2-3 MonoFusion-R iterations:

| Iterations | F1-all&darr; | MiD-err&darr; | Time (ms) |
| :---: | :---: | :---: | :---: |
| 1 (MonoFusion) | 4.53 | 44.8 | 176 |
| 2 | 4.40 | 42.2 | 268 |
| 3 | 4.33 | 40.0 | 377 |
| **4 (MonoFusion-R)** | **4.27** | **39.9** | 620 |
| TPCV (12 GRU iterations) | 4.68 | 41.8 | 784 |

*Inference speed measured on a single NVIDIA RTX 3090 at 352&times;1216. See the paper (Table 7) for the full ablation, including recurrent-loop initialization.*

## Repository Structure

The codebase mirrors the same layout for the main pipeline and the depth-supervised pretraining variant:

```
MonoFusion/
├── monofusion/                 # MonoFusion / MonoFusion-R model (used for OF / OE / TTC / scene flow on KITTI)
│   ├── model.py                #   top-level nn.Module (class MonoFusion)
│   ├── layers/                 #   shared building blocks: attention, matching, geometry, position, ...
│   └── network/                #   texture/scale encoders, fusion decoder, backbone
│       └── feature_net/        #     texture & scale transformer feature extractors
├── depth_supervised/            # auxiliary depth-supervised pretraining on FlyingThings3D (training-only depth head)
│   ├── monofusion_depth/        #   same layout as monofusion/, plus a disp_net.py depth head
│   ├── train.py / train.sh
│   └── submission.py / submission.sh
├── dataloader/                  # dataset loading & augmentation (FlyingChairs / FlyingThings3D / Driving / KITTI)
├── utils/                       # training loop (trainer.py), losses (loss.py), visualization (draw.py)
├── pretrained/                  # place downloaded checkpoints here
├── images/                      # figures used in this README
├── train.py / train.sh          # KITTI fine-tuning entry point
└── submission.py / submission.sh # inference / benchmark-submission entry point
```

> This structure replaces the original release's cryptic, experiment-suffixed names (e.g. `selfttc/scale_net_agg4gxr3/`, `self_ttc_agg4gxr3.py`) with names that describe what each module actually does. No model weights are affected: parameter names inside each `nn.Module` are unchanged, so previously-trained checkpoints still load correctly.

## Setup

### Environment

Experiments were run on Ubuntu 20.04, Python 3.8, PyTorch 1.12.0, CUDA 11.3, NVIDIA RTX 3090.

```bash
conda create -n monofusion python=3.8 -y
conda activate monofusion

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

```bash
git clone https://github.com/LChanglin/MonoFusion.git
cd MonoFusion
```

### Datasets

Download [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html), [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and [KITTI Scene Flow](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) and arrange them as:

```
data/
├── flyingthings/
│   ├── disparity/
│   ├── disparity_change/
│   ├── frames_cleanpass/
│   └── optical_flow/
└── kitti/
    ├── data_scene_flow/
    │   ├── training/
    │   └── testing/
    ├── data_scene_flow_calib/
    │   ├── training/
    │   └── testing/
    └── data_scene_flow_multi/
        ├── training/
        └── testing/
```

## Model Zoo

Download all checkpoints from [Google Drive](https://drive.google.com/drive/folders/1ebfjezbP6rDpzH5ofFLMWLl50YtE8-kT) into `./pretrained/` (see [`pretrained/README.md`](./pretrained/README.md) for a per-file description).

| Checkpoint | Model | F1-all&darr; | MiD-err&darr; |
| --- | --- | :---: | :---: |
| `monofusion_kitti.pth.tar` | MonoFusion, KITTI fine-tuned | 4.48 | 42.60 |
| `monofusion_r_kitti.pth.tar` | MonoFusion-R, KITTI fine-tuned | **4.27** | **39.86** |
| `monofusion_depth.pth.tar` | MonoFusion, depth-supervised, FlyingThings3D | 2.63 (EPE) | --- |

## Training

**1. Optical flow / scene flow.** Pretrain on FlyingChairs + FlyingThings3D (200k iters, lr 1e-4, batch size 4), then fine-tune on KITTI (50k iters, lr 5e-5, batch size 3):

```bash
# fine-tune on KITTI, starting from a FlyingThings3D-pretrained checkpoint
# --num_iterations 1 trains MonoFusion, --num_iterations 4 trains MonoFusion-R
sh train.sh
```

**2. Optical expansion.** Following prior work, first train on the [Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) dataset (100k iters, lr 1e-4), then fine-tune on KITTI (every 5th KITTI image held out for validation, 60k iters, lr 5e-5) by setting `--stage 'kitti'` (or `'driving'` for the first stage) in `train.sh`.

**3. Depth-supervised pretraining (optional, for FlyingThings3D generalization).** Because FlyingThings3D is a synthetic dataset with weak photometric static-scene cues, we provide an auxiliary, training-only depth-supervision scheme (the depth head is dropped again for inference/KITTI fine-tuning):

```bash
cd depth_supervised
sh train.sh
```

Key training flags (see `train.py --help` for the full list):

| Flag | Meaning |
| --- | --- |
| `--num_iterations` | number of MonoFusion-R recurrent iterations (`1` = MonoFusion) |
| `--resume` | checkpoint to initialize from (required unless `--load_flow_param` is set) |
| `--save` | output checkpoint path |
| `--stage` | training dataset: `chairs`, `things`, `driving`, `kitti`, `sintel`, ... (see `dataloader/dataset.py`) |
| `--image_size`, `--lr`, `--epoch`, `--batch_size` | standard training hyperparameters |

## Evaluation & Inference

```bash
# dumps KITTI-format flow/disparity predictions for benchmark submission
sh submission.sh
```

or directly:

```bash
CUDA_VISIBLE_DEVICES=0 python submission.py \
  --resume ./pretrained/monofusion_r_kitti.pth.tar \
  --num_iterations 4 \
  --inference_dir ./data/kitti/data_scene_flow/testing/
```

Results (flow `.png`, disparity `.png`, and colorized visualizations) are saved under `./output/`. Submit the KITTI scene-flow predictions to the [KITTI benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=scene_flow&result=2efaab1112242238d8dec77411fb10e2df6e0e36) for official numbers.

## Results

All numbers below are monocular unless noted otherwise; see the paper for full tables, per-module ablations, and the FlyingThings3D generalization study.

**Optical Flow (KITTI test set).** F1 = percentage of flow endpoint errors > 3px or > 5%.

| Method | Output | F1-all&darr; | F1-bg | F1-fg | Params (M) |
| --- | --- | :---: | :---: | :---: | :---: |
| RAFT3D (RGB-D) | Scene Flow | 4.29 | **3.39** | 8.79 | 51.3 |
| RAFT | Flow | 5.10 | 4.74 | 6.87 | 5.3 |
| GMA | Flow | 5.15 | 4.78 | 7.03 | 5.9 |
| FlowFormer | Flow | 4.68 | 4.37 | 6.18 | 18.2 |
| GMFlow+ | Flow | 4.49 | 4.27 | 5.60 | 4.7 |
| TPCV | Expansion & Flow | 4.69 | 4.53 | **5.52** | 10.8 |
| MonoFusion | Expansion & Flow | 4.48 | 4.14 | 6.14 | 17.9 |
| **MonoFusion-R** | Expansion & Flow | **4.27** | <ins>3.47</ins> | 5.97 | 21.1 |

**Optical Expansion (KITTI-40 validation set).** Input resolution 1242&times;375 on an RTX 3090.

| Method | Input | MiD-err&darr; | Time (s) |
| --- | --- | :---: | :---: |
| PRSM | Stereo | 124.0 | 3000 |
| OSF | Stereo | 115.0 | 300 |
| Yang & Ramanan | Mono | 75.00 | 0.21 |
| BiTTC | Mono | 73.55 | 2.2 |
| ScaleFlow | Mono | 48.90 | --- |
| TPCV | Mono | 42.80 | 0.74 |
| MonoFusion | Mono | 42.60 | **0.18** |
| **MonoFusion-R** | Mono | **39.86** | 0.62 |

**Time-to-Collision (KITTI).**

| Method | Err-1&darr; | Err-2&darr; | Err-5&darr; |
| --- | :---: | :---: | :---: |
| OSF | 1.79 | 2.93 | 4.03 |
| Yang & Ramanan | 1.86 | 1.87 | 2.45 |
| BiTTC | 2.40 | 3.16 | 3.25 |
| TPCV | 1.14 | 1.65 | **1.57** |
| MonoFusion | <ins>1.10</ins> | <ins>1.48</ins> | 1.75 |
| **MonoFusion-R** | **1.04** | **1.31** | <ins>1.68</ins> |

**Scene Flow (KITTI test set, monocular RGB methods).**

| Method | SF-all&darr; | D1-all&darr; | F1-all&darr; | SF-bg&darr; | SF-fg&darr; |
| --- | :---: | :---: | :---: | :---: | :---: |
| Yang & Ramanan | 8.12 | 1.81 | 6.30 | 7.06 | 13.44 |
| BiTTC | 8.50 | 1.81 | 6.31 | 7.45 | 13.74 |
| ScaleFlow | 6.94 | 1.81 | 5.32 | 6.06 | 11.32 |
| TPCV | 6.20 | 1.81 | 4.69 | 5.34 | 10.61 |
| MonoFusion | <ins>6.15</ins> | 1.81 | 4.48 | 5.04 | 11.70 |
| **MonoFusion-R** | **5.94** | 1.81 | 4.27 | 4.81 | 11.61 |

MonoFusion-R also stays competitive with SOTA **RGB-D** scene-flow methods despite using only a single monocular RGB stream (KITTI, non-occluded regions): SF-all **3.78** vs. 3.55 for CamLiFlow (M+D) and 4.53 for RAFT3D (M+D); MonoFusion-R even leads on F1-fg (3.10) and SF-fg (7.68). See the paper for the full occluded/non-occluded breakdown against RGB-D baselines.

**FlyingThings3D generalization (val/clean split, optical flow EPE).** MonoFusion, trained with the auxiliary depth-supervision scheme, generalizes to FlyingThings3D's unconstrained 6-DoF camera/object motion without any architecture change: **2.63** EPE vs. 2.66 for AnyFlow and 2.80 for GMFlow+ (full table in the paper's appendix).

## Visualization

KITTI optical flow / optical expansion qualitative results:

<p align="center">
  <img src="./images/viz.png" width="850" alt="KITTI qualitative results"/>
</p>

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026monofusion,
  title   = {Digging into Monocular Texture-Scale Feature Fusion for Optical Flow and Optical Expansion Estimation},
  author  = {Li, Changlin and Qian, Yeqiang and Wang, Chunxiang and Yang, Ming},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  note    = {early access}
}
```

## Acknowledgements

This code borrows components from [GMFlow](https://github.com/haofeixu/gmflow) and [RAFT](https://github.com/princeton-vl/RAFT). We thank the authors for open-sourcing their work.
