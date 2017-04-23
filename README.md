# kor-char-rnn-tensorflow
Korean language requires a little different treatment when we run character level RNN


multi-layer recurrent neural network for Korean (maybe applied to other asian language, including Japaneses and Chinese because there are so many distinct characters). 

There are a lot of toy example RNN for English, but there are few when we find a example using Korean language. 


한글 자연어처리에 있어서 딥러닝을 하는 경우 가장 먼저 해보게 되는게 글자단위의 RNN 모델일텐데, 이에 해당하는 자료가 별로 없어서 많은 사람들이 시작부터 어려움을 겪는 것 같습니다. 이 코드가 한글 자연어처리의 연구의 시작에 도움이 되었으면 합니다. 마음껏 수정하시고, 사용하시고 많이 배포하세요. (MIT Liscense)


Thanks to 

Sherjil Ozair's [char-rnn](https://github.com/sherjilozair/char-rnn-tensorflow)

Junyi Song (aka. socurites)'s [char-rnn-korean](https://github.com/socurites/char-rnn-korean)

Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)


## Requirements
- [Tensorflow 1.0](http://www.tensorflow.org)

## Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`. To access all the parameters use `python train.py --help`.

To sample from a checkpointed model, `python sample.py`.

## Datasets

한글을 훈련시키기 위한 데이터셋으로 1. 투명드래곤 소설 파일 2. 발라드 가사 모음 3. 나무위키 본문 을 사용할 수 있습니다.

특징은 다음과 같습니다. 

1. 투명드래곤 소설 파일:
	용량이 너무 작아서 (1 MB 미만) RNN이 의미있는 학습을 하기가 힘듭니다. 용량에 비해 문법에 어긋난 표현, 뜬금없는 표현들이 많이 나오기 때문에 많은 변수가 등장하지만, 이것을 학습할 정도로 반복적인 패턴이 데이터에 없으므로 train loss가 수많은 epoch이후에도 감소하지 않습니다. 고의로 모델을 복잡하게 만들어서 overfit 시키는 경우에는 사람이 볼 때 의미없는 결과가 출력됩니다. 

2. 발라드 가사 모음:
	용량이 적당합니다. (약 2MB). 많은 고유명사가 등장하지 않으며 대부분이 비슷한 문맥 (사랑, 이별)에 위치하면서 반복적인 패턴이 많이 등장합니다. 모델이 문법을 학습하는데 있어서 적당합니다. 어느정도 복잡한 모델 (hidden=700, layer=3, seq=100) 으로 약 900 epoch을 훈련시킨 이후에 train loss가 0.15 근처로 수렴하는 것을 볼 수 있습니다. (하지만 overfit을 한 느낌이 듭니다)


모델이 가사로써 봐줄만한 문법적, 의미적 오류범위 하에 생성하는것을 알 수 있습니다.
하지만 데이터셋의 가사를 약간 복사 붙여넣기 한다는 느낌이 있습니다. (overfit이 예상됨)
```
사랑 하는 것 위해서 
난 눈물이 나면 슬픔을 흔들어 

한 번쯤 다시 생각해 기다리겠어 
그대가 기억하는 나의 옛모습으로 
그러나 어느새 그대는 나를 잊었고 
내가 다가갈수록 그대는 멀어져 가네 
이렇게 쉽게 헤어질 우리였다면 
지난 긴 세월동안 그리워 이제는 
어둠에 깨져버린 우리 사랑을 
어떻게 살고 있는지 
저 멀리 그대 음성 
인사도 다른 어떤말도 못하고서 
그대 먼저 끊기만 기다려요 
어떤날은 잠에서 깨어난 
졸리운 목소리로
지나간 날들 모두 잊은 듯 
내마음 깊은 그대로 
사랑을 남긴 채
그대 내 맘에서 떠나가 버렸네 
아쉬움 남긴 채
내마음 깊은 그곳에 사랑을 
남긴 채
떠나가 버렸네 내맘속에 그대는
떠나가 버렸네 사랑했던 그대는

```

3. 나무위키 본문:
	데이터 자체가 위키 문법을 포함하고 있으므로 전처리를 거친 후에도 여전히 특이한 패턴들이 존재합니다. (항목을 나열하는 형태로 제작된 위키페이지 등) 학습을 위한 충분한 데이터가 있다고 판단되지만 (약 2GB) 각각의 주제와 서술 방식이 매우 다양하므로 이 모든것을 학습하려면 모델의 복잡도도 높아져야하고, 그 만큼 많은 데이터가 필요합니다. 이에 따라 필연적으로 훈련시간이 증가할 수 밖에 없습니다. 본 저자가 실험해본 결과 GTX 980 Ti (6GB VRAM) 에 fit 할 수 있는 최대 크기의 모델로 실험해도 10시간 동한 훈련해도 나아지지 않았습니다. 


You can use any plain text file as input. For example you could download [The complete Sherlock Holmes](https://sherlock-holm.es/ascii/) as such:

```bash
cd data
mkdir sherlock
cd sherlock
wget https://sherlock-holm.es/stories/plain-text/cnus.txt
mv cnus.txt input.txt
```

Then start train from the top level directory using `python train.py --data_dir=./data/sherlock/`

A quick tip to concatenate many small disparate `.txt` files into one large training file: `ls *.txt | xargs -L 1 cat >> input.txt`

## Tensorboard
To visualize training progress, model graphs, and internal state histograms:  fire up Tensorboard and point it at your `log_dir`.  E.g.:
```bash
$ tensorboard --logdir=./logs/
```

Then open a browser to [http://localhost:6006](http://localhost:6006) or the correct IP/Port specified.


## TODO


- [ ] add val loss calculation, and print it while training
- [ ] add how to get Korean dataset
- [ ] add how to reproduce my result


## Contributing
Please feel free to:
* Leave feedback in the issues
* Open a Pull Request
* Share your success stories and data sets!
