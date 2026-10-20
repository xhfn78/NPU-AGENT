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
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()


# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, 
#  value=[[ 0.8516406  -0.52920127 -0.9112464 ]]>,
#  <Variable path=sequential/dense/bias, shape=(3,),
#  dtype=float32, value=[0. 0. 0.]>,
#  <Variable path=sequential/dense_1/kernel,
#  shape=(3, 2), dtype=float32, value=[[ 0.6380346  -1.0862008 ]
print('===================model.weights=====================')
print(model.weights)
print('===================model.trainable weights=====================')
print(model.trainable_weights)
print(len(model.weights)) #6
print(len(model.trainable_weights)) #6


##########################동결############################
model.trainable= False  
##########################동결############################
print(len(model.weights)) #6
print(len(model.trainable_weights)) #0

model.summary()
# Total params: 17 (68.00 B)
#  Trainable params: 0 (0.00 B) #트레인 할수있는 파라미터 없어짐
#  Non-trainable params: 17 (68.00 B) #동결됨
