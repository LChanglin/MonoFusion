---
name: monofusion-reproduce
description: Download and verify MonoFusion checkpoints, run the reported four-iteration inference configuration with a single-pass control, and evaluate KITTI optical flow or historical visual references. Use when reproducing or troubleshooting LChanglin/MonoFusion results; distinguish compatibility checks from paper-score reproduction.
---

# MonoFusion reproduction

Reproduce the strongest *reported configuration* available in this repository and measure what the supplied checkpoint actually achieves. The default recipe is `legacy_scale`, four iterations, 352×1216, with a one-iteration control. Do not describe it as the measured best configuration unless evaluation on a specified held-out set supports that conclusion.

## Establish the experiment

Locate the user's MonoFusion checkout, Python environment, image-pair directory, and available GPU. An existing remote host may be used when the user authorizes it. Keep unrelated GPU jobs and shared environments intact. Record the repository commit, device, PyTorch/CUDA versions, checkpoint SHA-256, inference resolution, iterations, dataset split, and sample IDs with the outputs.

Read `pretrained/checkpoints.json`, `pretrained/README.md`, and the inference section of the root README in the checkout. The manifest pins the checked release. These two files have identical trained tensors; their names do not establish separate training runs. New files already carry `migration.checkpoint_arch`; do not reconvert them or use `strict=False` to hide a mismatch.

Ask for the dataset path only if it cannot be found from the task context. KITTI inputs must contain `image_2/000000_10.png` and its `_11.png` pair. A test-only dataset has no public flow ground truth; do not use colored reference predictions as truth. Establish a held-out split before comparing configurations. Training-set results on a potentially fine-tuned checkpoint must be labeled accordingly.

## Download and environment

From the repository root:

```bash
python skills/monofusion-reproduce/scripts/download_checkpoints.py --repo .
```

If this skill is installed separately, run its bundled `scripts/download_checkpoints.py` with `--repo /path/to/MonoFusion`. It downloads the two inference packages and checks size and SHA-256, reuses matching files, and refuses to overwrite mismatches. The current files are anonymously downloadable. If access or hash verification fails, retain the evidence and report the issue; do not relax the pinned checksum or repeatedly retry a changed release.

Use the root README's minimal inference environment; Python 3.13 / PyTorch 2.10.0+cu128 was GPU-validated. Only torch, numpy, Pillow and OpenCV are needed for this inference path. The native PyTorch attention fallback avoids building old `mmcv-full` for modern GPUs. Do not substitute zero outputs or disable attention to bypass an operator failure.

## Run the reported recipe

Use a fresh output directory for each configuration. Replace the example data path, output path and GPU index with the user's actual resources. Run a two-pair smoke test first using `--max_samples 2`, verify finite outputs and strict loading, then remove the limit for the agreed evaluation set.

```bash
CUDA_VISIBLE_DEVICES=0 python submission.py \
  --resume pretrained/monofusion_r_kitti.pth.tar \
  --num_iterations 4 --inference_size 352 1216 \
  --attn_splits_list 2 8 --corr_radius_list -1 4 --prop_radius_list -1 1 \
  --inference_dir /path/to/kitti/training --output_path output/reproduce/iter4

CUDA_VISIBLE_DEVICES=0 python submission.py \
  --resume pretrained/monofusion_kitti.pth.tar \
  --num_iterations 1 --inference_size 352 1216 \
  --attn_splits_list 2 8 --corr_radius_list -1 4 --prop_radius_list -1 1 \
  --inference_dir /path/to/kitti/training --output_path output/reproduce/iter1
```

`submission.py` reads the architecture metadata, disables TF32, resizes predictions back to input size, and emits raw NPZ plus flow PNG and visualizations. Validate `flow` as H×W×2 and `scale` as H×W, with finite values and positive scale. The two counts use identical tensors; more iterations are not guaranteed to improve every dataset. Keep a single parameter setting for all test images. Additional tuning belongs on a validation split, never on test references.

## Measure the result

With KITTI ground-truth flow PNGs, evaluate both output directories using the same IDs:

```bash
python skills/monofusion-reproduce/scripts/evaluate_flow.py \
  --predictions output/reproduce/iter4 \
  --ground-truth /path/to/kitti/training/flow_occ \
  --output output/reproduce/iter4_metrics.json
```

Use `--ids 000000 000005 ...` with an explicit full list when evaluating a held-out subset. Without `--ids`, the helper requires predictions for every GT file and fails on missing ones. It checks uint16 KITTI decoding and the validity mask, evaluates original-resolution raw flow, and reports pixel-weighted EPE and Fl-all plus image-mean values. The outlier condition is **EPE > 3 px AND EPE > 5% of ground-truth flow magnitude**. See the [official KITTI definition](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow). Local labeled-split scores are not official KITTI test scores. This helper does not evaluate MiD, TTC or scene flow.

If only `monofusion_c_kittc.zip` is available, use `migration/prepare_reference_samples.py` and `migration/validate.py` with the downloaded checkpoint, `--iterations 1` / `4`, `--size 352 1216`, and `--limit 40`. Report RGB MAE strictly as historical-image agreement. Existing compatibility results are documented in `migration/USAGE.txt`; exact checkpoint identity links the downloaded release to those tests. Keep both representative and worst-case comparisons. If neither GT nor historical references are available, finish an inference smoke test and report that quality is unverified.

Full scene-flow export needs external `disp_ganet_testing/*_10.png`; do not invent missing disparity. Official benchmark submission, publishing results, or changing repository data is a separate action requiring the user's applicable authorization.

## Deliver

Provide the commands, environment and checkpoint provenance, number and identity of evaluated pairs, metrics for both counts, raw predictions, representative and worst-case comparisons, and a short conclusion. State whether four iterations actually improved the chosen metric. Separate completed compatibility/visual checks, local ground-truth evaluation, and unavailable official paper-score reproduction. Do not claim the paper's 4.27 F1-all / 39.86 MiD solely from successful loading or an attractive visualization.
