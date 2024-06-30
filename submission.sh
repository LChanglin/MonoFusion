# CUDA_VISIBLE_DEVICES=1 python train.py \
# --resume ./pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --reg_refine \
# --num_reg_refine 6 \
# --batch_size 1 \
# --traindataset /mnt/pool2/lcl/data/data_scene_flow/training/ 

# --resume ./pretrained/cyclesma_xr3_kitti2.pth.tar \

CUDA_VISIBLE_DEVICES=3 python submission.py \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--num_head 1 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--inference_dir /home/lcl/data/testing/


# --inference_dir ~/data/18/

# --inference_dir ~/data/kitti/data_scene_flow_multi/training/image_2/
# --inference_dir /mnt/pool2/lcl/data/LB_full/LB_full/
# --inference_dir /mnt/pool2/lcl/data/kitti/data_scene_flow_multi/training/image_2/
# --resume ./pretrained/gmflowx_driving2kitti80y.pth.tar \

# --reg_refine \
# --num_reg_refine 6 \
# --resume ./pretrained/gmflowx_kitti_0522x.pth.tar \

# CUDA_VISIBLE_DEVICES=0 python test.py \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --reg_refine \
# --num_reg_refine 6 \
# --resume ./pretrained/gmflow_refine_driving_kitti.pth.tar \
# --inference_dir /mnt/pool2/lcl/data/kitti/data_scene_flow_multi/training/image_2/