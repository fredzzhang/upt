CKPT_DIR=checkpoints/
for entry in "$CKPT_DIR"/*
do
    CUDA_VISIBLE_DEVICES=0 python main.py --eval --port 2345 --resume $entry &>log/e${entry: -5:2} & PID=$!
    wait $PID
    echo "${entry: -5:2} done"
done
