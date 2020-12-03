CUDA_VISIBLE_DEVICES=3 \
python -u transmem.py \
--data-dir /home/chenpeixian/reid/dataset/ \
-a DDAN \
-b 64 \
--height 256 \
--width 128 \
--logs-dir ./logs/ \
--epoch 100 \
--workers=4 \
--lr 0.1    \
--num-instance 4 \
--tri-weight 1 \
--margin 0.3 \
--adv-weight 0.18 \
--mem-weight 0.05 \
--knn 8 \
--beta 0.002 \
--alpha 0.05 \
--features 1280 \
--dropout 0.5 \
--seed 0 \
# --resume ./logs/checkpoint-100.pth.tar \
# --evaluate \

