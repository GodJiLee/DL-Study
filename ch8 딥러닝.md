# 8. 딥러닝

## 8.1 더 깊게

##### 지금까지 배운 딥러닝의 여러 기술들(CNN, 매개변수 최적화, 계층)을 집약하여 MNIST 손글씨 데이터셋에 대한 딥러닝 학습 진행

### 8.1.1 더 깊은 신경망으로 

##### 그림 8-1의 경우 VGG를 사용한 CNN계층으로,
* 3 x 3의 작은 필터를 사용하는 Conv 계층
* ReLu 활성화 함수 사용
* 완전연결계층 뒤 드롭아웃 계층
* Adam
* He초깃값 사용 (비선형함수)

> 해당 신경망의 경우 정확도가 99.38%로 효과적인 학습을 진행함 
##### https://github.com/WegraLee/deep-learning-from-scratch/tree/master/ch08


```python
  import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
os.chdir("C://Users//leejiwon//Desktop//프로그래밍//deep//deep-learning-from-scratch-master//deep-learning-from-scratch-master//dataset")
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    """정확도 99% 이상의 고정밀 합성곱 신경망
    네트워크 구성은 아래와 같음
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10):
        # 가중치 초기화===========
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값
        
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx+1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size) 

        # 계층 생성===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], 
                           conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], 
                           conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                           conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                           conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                           conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))
        
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]
```

##### 결과는 그림 8-2와 같이 인간의 인식 오류와 비슷한 수준의 학습이 진행되었음
> 인공지능의 잠재력이 대단하다는 것을 알 수 있음

### 8.1.2 정확도를 더 높이려면 

* 손글씨 데이터 딥러닝 학습 정확도 순위 
https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

* 손글씨 데이터의 경우 비교적 단순한 데이터이므로 깊은 신경망계층이 필요하지 않음
> 목록에 소개된 신경망 역시 합성곱 2계층과 완전연결 2계층으로 이루어진 단순한 형태

#### 더 정확한 분석을 위해서는 한 층 더 고차원적인 분석기법을 적용하면 됨
> 데이터 확장, 앙상블 학습, 학습률 감소 등 

 데이터 확장 : 기존 데이터들에 대한 회전, 이동을 통해 인위적으로 데이터의 크기를 늘리는 작업
> 기존 데이터가 적은 경우 유용함
>> 데이터 확장은 데이터 변형 이외에도 1) crop (잘라내기) 2) flip (좌우반전 : 대칭 이미지에 대해서만 적용 가능) 3) 외형 변화 (밝기) 4) 스케일 변화 (확대, 축소) 등의 다양한 방법이 존재함

### 8.1.3 깊게 하는 이유

##### 층을 깊게 하는 것의 중요성에 대한 과학적인 증명은 존재하지 않음
> ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 에서 진행되는 대규모 이미지 인식 경진대회에서 대부분 층의 깊이와 정확도가 비례한다는 점에서 그 이점을 확인할 수 있음

* 층을 깊게 하는 것의 이점
##### 1) 신경망의 매개변수 수가 줄어듦 : 매개변수별 표현력이 증가함
> 그림 8-5, 8-6을 참고하여 5 x 5 합성곱 한 층일 때의 뉴런 수 (25개) > 3 x 3 합성곱 두 층일 때의 뉴런 수 (18개) 이라는 점을 확인 가능
>> 매개변수 수가 줄어듦에 따라 더 넓은 수용 영역을 확보할 수 있으며 비선형 활성화 함수가 층을 거듭할 수록 추가되므로 더 복잡한 표현이 가능함
##### 2) 학습의 효율성이 증가함
> CNN은 학습을 계층적으로 수행하므로 학습 데이터 양을 줄이며 효율적인 학습을 가능토록 함
>> 1) 계층적 분해 (분업화) 2) 계층적 전달 (문제 할당) 
>>> 개 사진을 한 번에 인식하는 것보다 여러 계층을 거쳐 단계별로 이해하는 것이 더 효율적
