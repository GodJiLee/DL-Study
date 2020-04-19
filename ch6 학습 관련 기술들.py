#!/usr/bin/env python
# coding: utf-8

# Chapter 6 학습 관련 기술들

# ## 6.1 매개변수 갱신

# ##### 손실함수를 최소화하는 과정인 "최적화" 
# ##### 실제로는 매개변수의 공간이 넓고 복잡하기 때문에 순식간에 최소값을 찾는 일은 불가능함
# ##### SGD : 확률적 경사 하강법은 매개변수의 기울기를 이용해서 최소값을 찾는 방법
# 이보다 더 효율적인 방법도 존재

# ## 6.1.1 모험가 이야기

# 손실함수의 최솟값을 찾는 문제를 '깊은 산골짜기를 탐험하는 모험가'에 비유함

# ## 6.1.2 확률적 경사 하강법(SGD)

# In[ ]:


import numpy as np


# In[ ]:


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# In[ ]:


class SGD: 
    def __ init__(self, lr = 0.01):
        self.lr = lr
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# In[ ]:





# In[ ]:




