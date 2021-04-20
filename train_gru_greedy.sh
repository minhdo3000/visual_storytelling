#!/bin/sh
echo 'Start training'
echo 'Encoder = Bi-GRU, decoder = GRU, beam size = 1'

CUDA_VISIBLE_DEVICES=0 python train.py --id XE --data_dir  data/rl_story/data --start_rl -1 --rnn_type gru --beam_size 1


CUDA_VISIBLE_DEVICES=0 python train_rl_story.py --id rl_story --start_from_model data/save/XE/gru/model.pth --rnn_type gru --data_dir  data/rl_story/data --beam_size 1
echo 'End training'
