# Pretrained Weights

Download all pretrained checkpoints from Google Drive and place them in this folder:

https://drive.google.com/drive/folders/1ebfjezbP6rDpzH5ofFLMWLl50YtE8-kT

| File | Model | Trained on | Notes |
| --- | --- | --- | --- |
| `monofusion_depth.pth.tar` | MonoFusion (+ auxiliary depth head) | FlyingThings3D, depth-supervised (`depth_supervised/train.sh`) | SOTA optical-flow generalization on FlyingThings3D; also used to initialize KITTI fine-tuning (`train.sh --resume`) |
| `monofusion_kitti.pth.tar` | MonoFusion (`--num_iterations 1`) | KITTI | Single fusion-decode pass |
| `monofusion_r_kitti.pth.tar` | MonoFusion-R (`--num_iterations 4`) | KITTI | Recurrent texture-scale refinement, best accuracy |

See the [main README](../README.md#model-zoo) for the corresponding benchmark numbers.
