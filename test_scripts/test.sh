#----------------------------------------TTA-------------------------------------
CUDA_VISIBLE_DEVICES=1  python -u test_TTA_so.py  --dataset dior --net res50 --cuda --load_name /data/liuweixing/DA/training/SF/res50/dior/SFso_False_target_clipart_gamma_5_1_7_23449.pth
#CUDA_VISIBLE_DEVICES=3  python -u test_TTA_NORM.py  --dataset dior --net res50 --cuda --load_name /data/liuweixing/DA/training/SF/res50/dior/SFso_False_target_clipart_gamma_5_1_7_23449.pth
#CUDA_VISIBLE_DEVICES=3  python -u test_TTA_DUA.py  --dataset dior --net res50 --cuda --load_name /data/liuweixing/DA/training/SF/res50/dior/SFso_False_target_clipart_gamma_5_1_7_23449.pth
#CUDA_VISIBLE_DEVICES=3  python -u test_TTA_ActMAD.py  --dataset dior --net res50 --cuda --load_name /data/liuweixing/DA/training/SF/res50/dior/SFso_False_target_clipart_gamma_5_1_7_23449.pth
#CUDA_VISIBLE_DEVICES=1  python -u test_TTA_ours.py --lr 3e-4  --dataset dior --net res50 --cuda --load_name /data/liuweixing/DA/training/SF/res50/dior/SFso_False_target_clipart_gamma_5_1_7_23449.pth
