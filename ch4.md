## 4.2.3 미니배치 학습


```python
import sys, os #시스템, 운영체제와 상호작용하는 파이썬 함수as
sys.path.append(os.pardir) #부모 경로 지정
os.chdir('C://Users//leejiwon//Desktop//deep-learning-from-scratch-master//deep-learning-from-scratch-master//dataset')
import numpy as np #넘파이 불러오기
from dataset.mnist import load_mnist #mnist 데이터셋에서 load_mnist 불러오기
```


```python
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True) #mnist 데이터 읽어오기 #원-핫인코딩으로 가져옴
```


```python
print(x_train.shape) #60000개의 훈련 데이터, 784개의 입력
```

    (60000, 784)
    


```python
print(t_train.shape) #60000개의 데이터, 10개의 정답 레이블
```

    (60000, 10)
    


```python
#미니배치로 10장만 빼내기
train_size = x_train.shape[0] #60000개의 훈련데이터
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #(범0위, 뽑을 개수)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```


```python
np.random.choice(60000, 10)
```




    array([54615, 37091, 12089, 21613, 20550, 45668,  9283, 29723, 34712,
           31705])



## 4.2.4 (배치용) 교차 엔트로피 오차 구현하기


```python
#데이터를 하나씩 or 배치로 처리 가능
def cross_entropy_error(y, t):eee
    if y.ndim == 1: #y가 1차원의 데이터일 때 (데이터 배치 처리x, 하나씩 처리 o) 
        t = t.reshape(1, t.size) #형상을 바꿔줌 #정답 레이블의 뉴련 수 
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0] #새로 정의한 y에 대한
    return -np.sum(t * np.log(y + 1e-7)) / batch_size #배치의 크기로 나눠 정규화 > 이미지 1장당 오차 계산
```


```python
#정답 레이블이 원-핫 인코딩이 아닐 때
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) 
        y = y.reshape(1, y.size)
        
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #정답이 아닌 값들도 계산에 포함됨
                                #np.arnage(batch_size) : 0 ~ batch_size-1까지의 배열 생성 
                                #인덱스이므로 batch_size가 5이면 [0,1,2,3,4]
                                #y[0,2], y[1,7], y[2,0], ...
```

## 4.2.5 왜 손실 함수를 정의하는가?

#### 손실함수는 정확도의 우회적인 값
 * 굳이 손실함수를 사용하는 것은 손실함수 최소화 문제에 함수 미분을 사용할 수 있기 때문

#### 미분 : 값을 아주 조금 변화시킬 때 함수값의 변화 
 * == 기울기 (음수, 양수에 따라 갱신 방향을 결정할 수 있음)

#### 정확도는 매개변수의 미분이 대부분의 장소에서 0이 되기 때문에 학습(갱신)이 불가능함
 * 정확도는 34% 와 같이 불연속적인 값으로 계산되며 약간의 조정으로는 정확도 개선이 일어나지 않음
 * 계단함수를 활성화함수로 사용하지 않는 이유와 같음 (대부분의 region에서 기울기 = 0)

# 4.3 수치 미분 

#### 경사법 (기울기를 기준으로 갱신 방향 설정)

## 4.3.1 미분

#### 평균 변화량과 순간 변화량 
 * 미분은 순간 변화량 
 
#### [식 4.4] : x의 작은 변화가 f(x)를 얼마나 변화시키는지


```python
# 잘못된 구현
def numerical_diff(f, x): #수치미분 : 아주 작은 차분으로 미분하는 것
    h = 10e-50 #작은 값 대입 #0.000....1 
    return (f(x+h) - f(x)) / h
```


```python
#한계 1) 반올림 오차 : 소수점 아래 8자리 수는 생략해버림
np.float32(1e-50) #32비트 부동 소수점 
```




    0.0




```python
#한계 2) 차분 오차 : 작은 값이지만 결과적으로는 평균변화율 
#중앙 차분(중심 차분)으로 해결 : x+h와 x-h
def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h)) / (2*h)
```

## 4.3.2 수치 미분의 예

#### y = 0.01 * x^2 + 0.1 * x 미분


```python
def function_1(x): 
    return 0.01*x**2 + 0.1*x #함수 정의
```


```python
import numpy as np
import matplotlib.pylab as plt #시각화 패키지 불러옴

x = np.arange(0.0, 20.0, 0.1) #0~19.9까지 0.1의 간격의 배열 x 생성
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)") #라벨링
plt.plot(x, y)
plt.show()
```


![png](output_22_0.png)



```python
numerical_diff(function_1, 5) # x = 5 일 때 수치 미분 #2에 가까움
```




    0.1999999999990898




```python
numerical_diff(function_1, 10) #3에 가까움 #각각은 함수의 기울기 값
```




    0.2999999999986347




```python
def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y

def draw(ax, x, y, line, tox, toy):
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.plot(x, y)
    ax.plot(x, line)
    h = np.arange(0, tox, 0.01)
    v = np.arange(-1, toy, 0.01)
    ax.plot(np.array([tox for _ in range(v.size)]), v, 'k--')
    ax.plot(h, np.array([toy for _ in range(h.size)]), 'k--')
    ax.scatter(tox,toy,color='red')
    ax.set_xlim([0,20])

tf = tangent_line(function_1, 5)
y2 = tf(x)
tf = tangent_line(function_1, 10)
y3 = tf(x)

f, (ax1, ax2) = plt.subplots(2, 1)
draw(ax1, x, y, y2, 5, function_1(5))
draw(ax2, x, y, y3, 10, function_1(10))
```


![png](output_25_0.png)


## 4.3.3 편미분

#### 변수가 2개인 함수


```python
def function_2(x):
    return x[0]**2 + x[1]**2 #각각의 인수를 넘파이 배열로 간주
```


```python
from mpl_toolkits.mplot3d import Axes3D
X = np.arange(-3, 3, 0.25) #x값 정의
Y = np.arange(-3, 3, 0.25) #y값 정의
XX, YY = np.meshgrid(X, Y) 
ZZ = XX**2 + YY**2 #함수식

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='hot'); #변수 두 개의 함수를 시각화
```


![png](output_29_0.png)



```python
# [식 4.6] 미분 > 편미분 : 변수가 여럿인 함수에 대한 미분
# x0 = 3, x1 = 4일 때, x0에 대한 편미분
def function_tmp1(x0): 
    return x0 * x0 + 4.0 ** 2.0 #미분 대상이 아닌 변수는 상수 취급 
                            # 특정 값 고정 목표함수 재정의

numerical_diff(function_tmp1, 3.0) #수치 미분 실행
```




    6.00000000000378




```python
#x0 = 3, x1 = 4일 때, x1에 대한 편미분
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1 

numerical_diff(function_tmp2, 4.0) 
```




    7.999999999999119



# 4.4 기울기


```python
import numpy as np

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #첫 번째 매개변수이므로 입력층, 은닉층 1에 대한 뉴런 수 적용
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) #두 번째 매개변수이므로 은닉층, 출력층에 대한 뉴런 수 적용
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1 #예측함수
        z1 = sigmoid(a1) #시그모이드에 적용 (활성화함수)
        a2 = np.dot(z1, W2) + b2 #2번째 예측함수
        y = softmax(a2) #소프트맥스에 적용 (확률값)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t): 
        y = self.predict(x) #손실함수
        
        return cross_entropy_error(y, t) #교차 엔트로피 오차 
    
    def accuracy(self, x, t): #정확도 
        y = self.predict(x)
        y = np.argmax(y, axis=1)                             
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
```


```python

```
