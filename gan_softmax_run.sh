CUDA_VISIBLE_DEVICES=0 python gan_softmax_loss.py -a densenet121 -b 32 --height 288 --width 112 \
         --lr 0.0001 --weight-decay 0.001 --step_size 26 --decay_step 30 \
         --epochs 110 --start_save 1 --print-info 20 --combine-trainval \
        --pretrained_model /home/shenxu.sx/lyj/code/epoch_124.pth.tar \
         --logs-dir /home/shenxu.sx/lyj/logs/reid_semi_softmax \
         --outf /home/shenxu.sx/lyj/data/reid_semi_softmax  \
         --cuda | tee /home/shenxu.sx/lyj/logs/reid_semi_softmax2.txt

