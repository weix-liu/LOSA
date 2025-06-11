#---------------------TTA-----------------
CUDA_VISIBLE_DEVICES=1  python -u trainval_SFso.py   --cuda --lr 0.001  --net res101  --dataset corsadd   --save_dir TTAtraining/
#CUDA_VISIBLE_DEVICES=1  python -u trainval_SFso.py   --cuda --lr 0.001  --net res101  --dataset xview   --save_dir TTAtraining/
#CUDA_VISIBLE_DEVICES=0  python -u trainval_SFso.py   --cuda --lr 0.001  --net res101  --dataset itcvd   --save_dir TTAtraining/
#CUDA_VISIBLE_DEVICES=1  python -u trainval_SFso.py   --cuda --lr 0.001  --net res50  --dataset nwpu10   --save_dir TTAtraining/
#CUDA_VISIBLE_DEVICES=2  python -u trainval_SFso.py   --cuda --lr 0.001  --net res50  --dataset dior   --save_dir TTAtraining/
#CUDA_VISIBLE_DEVICES=1  python -u trainval_SFso.py   --cuda --lr 0.001  --net res101  --dataset pascal_voc_water  --save_dir TTAtraining/