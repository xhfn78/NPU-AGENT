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
x = np.array([1,])
y = np.array([1,])

#2모델
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()



##########################동결############################
# model.trainable= False  
model.trainable= True  

##########################동결############################

# print('=======================================')
# print(model.weights)
# print('=======================================')



#3.컴파일  훈련
model.compile(loss= 'mse', optimizer = 'adam')
model.fit (x,y ,batch_size=1, epochs=500, verbose=1)


#4 평가예측

y_predict  = model.predict(x)
print(y_predict)

print('=======================================')
print(model.weights)
print('=======================================')
