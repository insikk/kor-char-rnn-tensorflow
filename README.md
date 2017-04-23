# kor-char-rnn-tensorflow
Korean language requires a little different treatment when we run character level RNN


multi-layer recurrent neural network for Korean (maybe applied to other asian language, including Japaneses and Chinese because there are so many distinct characters). 

There are a lot of toy example RNN for English, but there are few when we find a example using Korean language. 


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
