
## Prerequisites 
- Python 2.7
- PyTorch 0.3
- TensorFlow (optional, only using the fantastic tensorboard)
- cuda & cudnn

## Usage
### 1. Setup

Download the preprocessed ResNet-152 features [here](https://vist-rl_story.s3.amazonaws.com/resnet_features.zip) and unzip it into `data/resnet_features`.

### 2. Generate Story using Reinforcemnt Learning
To train an the model:
- Encoder-decoder: Bi-GRU and GRU, greedy, run: `./train_gru_greedy.sh`
- Encoder-decoder: Bi-GRU and GRU, beam size 2, run: `./train_gru_beam.sh`
- Encoder-decoder: Bi-LSTM and LSTM, greedy, run: `./train_lstm_greedy.sh`
- Encoder-decoder: Bi-LSTM and LSTM, beam size 2, run: `./train_lstm_beam.sh`
- Encoder-decoder: Bi-GRU with Luong attention and GRU, greedy, run: `./train_gru_attn_greedy.sh`
- Encoder-decoder: Bi-GRU with Luong attention and GRU, beam size 2,  run: `./train_gru_attn_beam.sh`

Note that `PRETRAINED_MODEL` can be `data/save/XE/model.pth` or some other saved models. 
Check `opt.py` for more information.

### 4. Monitor your training
TensorBoard is used to monitor the training process. Suppose you set the option `checkpoint_path` as `data/save`, then run
`tensorboard --logdir data/save/tensorboard`

## Reference

https://github.com/eric-xw/rl_story
