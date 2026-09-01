import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
#1. 데이터
# x = np.array([[1,2,3,4,5],   # x.shape :  (5, 1)
#               [6,7,8,9,10]])
x = np.array([[1.6],[2.7],[3.8],[4.9],[5.10]])
y = np.array([1,2,3,4,5])

print("x.shape : ", x.shape) #x.shape :  (5, 1)
print("y.shape : ", y.shape) #y.shape :  (5,)
#2. 모델 구성

#3.컴파일, 훈련

#4.평가예측