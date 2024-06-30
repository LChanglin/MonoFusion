CUDA_VISIBLE_DEVICES=3 python submission.py \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--num_head 1 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--inference_dir /home/lcl/data/testing/