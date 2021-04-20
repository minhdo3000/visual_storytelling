#!/bin/sh
echo 'Start training'
echo 'Encoder = Bi-GRU with attention, decoder = GRU, beam size = 2'

CUDA_VISIBLE_DEVICES=0 python train.py --id XE --data_dir data/rl_story/data  --start_rl -1 --rnn_type gru_att --weight_decay 0.00001 --beam_size 2

CUDA_VISIBLE_DEVICES=0 python train_rl_story.py --id rl_story --start_from_model data/save/XE/gru_att/model.pth --rnn_type gru_att --data_dir /home/kangjie/PycharmProjects/rl_story/data --save_checkpoint_every 100  --workers 8 --max_epochs 20 --beam_size 2  
echo 'End training'


