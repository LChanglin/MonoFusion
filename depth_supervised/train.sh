OMP_NUM_THREADS=8 torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
--load_flow_param \
--save "${MONOFUSION_CHECKPOINT:?Set a checkpoint path for this model}" \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--num_iterations 1 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--lr 2e-5 \
--epoch 8 \
--batch_size 1 \
--stage 'things' \
--image_size 384 768 \
--parallel \
# --load_opt
