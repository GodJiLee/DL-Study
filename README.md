Machine leanring 2주차
chapter. 인공신경망 2 XOR
import numpy as np
import matplotlib.pyplot as plt

#AND 게이트
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.2, 0.2])
    b = -30
    a = np.sum(x * w) + b
    if a <= 0:
        return 0
    else:
        return 1
[AND(100, 100), AND(100, -100), AND(-100, 100), AND(-100, -100)]
[1, 0, 0, 0]
#OR 게이트
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.05, 0.05])
    b = 0
    a = np.sum(x * w) + b
    if a < 0:
        return 0
    else:
        return 1
[OR(100, 100), OR(100, -100), OR(-100, 100), OR(-100, -100)]
[1, 1, 1, 0]
#NAND 게이트
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.2, -0.2])
    b = 30
    a = np.sum(x * w) + b
    if a <= 0:
        return 0
    else:
        return 1
[NAND(100, 100), NAND(100, -100), NAND(-100, 100), NAND(-100, -100)]
[0, 1, 1, 1]
#XOR 게이트
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
[XOR(100, 100), XOR(100, -100), XOR(-100, 100), XOR(-100, -100)]
[0, 1, 1, 0]
#AND 게이트
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([20, 20])
    b = -30
    a = np.sum(x * w) + b
    if a <= 0:
        return -1
    else:
        return 1
#OR 게이트
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([20, 20])
    b = -10
    a = np.sum(x * w) + b
    if a <= 0:
        return 0
    else:
        return 1
#NAND 게이트
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-20, -20])
    b = 30
    a = np.sum(x * w) + b
    if a <= 0:
        return 0
    else:
        return 1
#XOR 게이트
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
[XOR(1, 1), XOR(1, 0), XOR(0, 1), XOR(0, 0)]
[-1, 1, 1, -1]
#계단함수
def step_function(a):
    if a > 0:
        return 1
    else:
        return 0
    # y > 0
    # return y.astype(np.int)
import numpy as np
x = np.array([-1.0, 1.0, 2.0])
x
array([-1.,  1.,  2.])
y = x > 0; y.astype(as.int)
x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, step_function(x))
plt.ylim(-0.1, 1.1) #y축의 범위 지정
plt.show()

def relu(a):
    return np.maximum(0, a)
relu(x)
array([0., 1., 2.])
def sigmoid(a):
    return 1 / (1 + np.exp(-a))
​
x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)
array([0.26894142, 0.73105858, 0.88079708])
x = np.arange(-5.0, 5.0, 0.1) #-5부터 5까지 0.1의 간격으로 값을 잡음
plt.plot(x, sigmoid(x), c = 'r')
plt.plot(x, relu(x), c = 'b')
plt.ylim(-0.1, 1.1) #y축 범위 지정
plt.show()

A = np.array([1, 2, 3, 4])
print(A)
[1 2 3 4]
B = np.array([[1, 2], [3, 4]])
print(B)
[[1 2]
 [3 4]]
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
def step_function(x):
    y = x > 0
    return y.astype(np.int)
​
x = np.array([-1.0, 1.0, 2.0]) #numpy 배열 생성
x
array([-1.,  1.,  2.])
y = x > 0
y
array([False,  True,  True])
def step_function(x):
    return np.array(x > 0, dtype = np.int) #0보다 크면 TRUE, 아니면 FALSE 반환 (TRUE = 1, FASLE = 0)
​
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y축의 범위 지정
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
​
x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)
array([0.26894142, 0.73105858, 0.88079708])
t = np.array([-1.0, 2.0, 3.0])
1.0 + t
array([0., 3., 4.])
1.0 / t
array([-1.        ,  0.5       ,  0.33333333])
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y축 범위 지정
plt.show()
