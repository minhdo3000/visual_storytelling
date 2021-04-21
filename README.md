## Visual Story Generation  using Adversarial Learning
Generating natural language stories from images has become a major research area in the field of computer vision. Though there are impressive results in visual story generation, the task of producing more human-like stories is still restrictive. Inspired by existing reinforcement reward learning methods that aim to cope up with this challenge, we consider adapting them to boost visual story generation tasks. In this project, we work on [VIST](https://vist-rl_story.s3.amazonaws.com/resnet_features.zip), the Visual Storytelling dataset and perform experiments on different RNN variants along with adversarial learning, that aims at generating coherent and expressive stories for image sequences. While we empirically show the superiority of our proposed method with visual attention mechanism in adversarial setup over the state-of-art methods, we also mention difficulties due to the limitations of automatic metrics on story evaluation. 

## Prerequisites 
- Python 2.7
- PyTorch 0.3
- TensorFlow (optional, only using the fantastic tensorboard)
- cuda & cudnn

## Usage
### 1. Setup
- Create anaconda environment: `conda create -n rl_story python=2.7`
- Install pytorch: `conda install pytorch=0.4.1 cuda90 -c pytorch`
- Install tensorflow: 'conda install tensorflow=1.15'
- Download the code: `git clone https://github.com/minhdo3000/visual_storytelling.git`
- Download the preprocessed ResNet-152 features [here](https://vist-rl_story.s3.amazonaws.com/resnet_features.zip) and unzip it into `data/resnet_features`.

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
