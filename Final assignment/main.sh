wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 4 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "efficient-net-b2-training, 320X227  RMS prop,batch_size=4, DiceLoss" \
