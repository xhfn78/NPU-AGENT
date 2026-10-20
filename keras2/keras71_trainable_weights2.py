import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__)

np.random.seed(333)
tf.random.set_seed(333)
random.seed(333)


#1데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2모델
model = Sequential()
model.add(Dense(16,input_dim=1))
model.add(Dense(8))
model.add(Dense(1))
model.summary()


##########################동결############################
model.trainable= False  
# model.trainable= True  

##########################동결############################

#3.컴파일  훈련
model.compile(loss= 'mse', optimizer = 'adam')
model.fit (x,y ,batch_size=1, epochs=100, verbose=1)


#4 평가예측

y_predict  = model.predict(x)
print(y_predict)
# 동결 x
# [[1.1344696]
#  [2.0827157]
#  [3.0309618]
#  [3.9792075]
#  [4.927453 ]]


# #동결 o
# [[-0.17050676]
#  [-0.34101352]
#  [-0.5115203 ]
#  [-0.68202704]
#  [-0.8525342 ]]