#!/bin/sh
echo 'Start training'
echo 'Encoder = Bi-LSTM, decoder = LSTM, beam size = 2'

CUDA_VISIBLE_DEVICES=0 python train.py --id XE --data_dir  data/rl_story/data --start_rl -1 --rnn_type lstm --beam_size 2

CUDA_VISIBLE_DEVICES=0 python train_rl_story.py --id rl_story --start_from_model data/save/XE/lstm/model.pth --rnn_type lstm --data_dir data/rl_story/data --save_checkpoint_every 500  --workers 8 --max_epochs 20 --beam_size 2

echo 'End training'


