# Fine-tune on KITTI, starting from a FlyingChairs/FlyingThings3D pretrained checkpoint.
# --num_iterations 1 trains MonoFusion (single fusion-decode pass)
# --num_iterations 4 trains MonoFusion-R (recurrent refinement, used for our best results)
OMP_NUM_THREADS=8 torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
--resume ./pretrained/monofusion_depth.pth.tar \
--save ./pretrained/monofusion_r_kitti.pth.tar \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--num_iterations 4 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--lr 8e-5 \
--epoch 10 \
--batch_size 1 \
--stage 'kitti' \
--image_size 384 768 \
--parallel \
# --load_opt
