# Run inference on KITTI and dump results (flow / disparity) for benchmark submission.
# See README.md "Evaluation" section for details.
#
# --num_iterations 1 reproduces MonoFusion (single fusion-decode pass)
# --num_iterations 4 reproduces MonoFusion-R (recurrent refinement, used for our
#                     best reported numbers)

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
