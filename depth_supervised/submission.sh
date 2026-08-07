# Run the depth-supervised model (used for the FlyingThings3D generalization
# results, see README.md) on a directory of test images.
CUDA_VISIBLE_DEVICES=0 python submission.py \
--resume ./pretrained/monofusion_depth.pth.tar \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--num_head 1 \
--num_iterations 1 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--inference_dir ./data/flyingthings/testing/
