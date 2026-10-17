# Run inference on KITTI and dump results (flow / disparity) for benchmark submission.
# See README.md "Evaluation" section for details.
#
# Download the updated checkpoint from the README link; no conversion is needed.
# --num_iterations 1 runs a single fusion-decode pass.
# --num_iterations 4 runs recurrent refinement. Compatibility validation does not
# establish that the paper benchmark scores have been reproduced.

CUDA_VISIBLE_DEVICES=0 python submission.py \
--resume ./pretrained/monofusion_r_kitti.pth.tar \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--num_head 1 \
--num_iterations 4 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--inference_dir ./data/kitti/data_scene_flow/testing/
