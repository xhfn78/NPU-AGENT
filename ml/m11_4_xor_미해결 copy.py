import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)


#1데이터  
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])

#모델
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim=2,activation='sigmoid'))


#3컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam',metrics =['acc'])
model.fit(x_data,y_data, epochs =100)


#4평가 예측

y_predict =model.predict(x_data)

loss = model.evaluate(x_data,y_data)
print('loss,acc:',loss)

acc = accuracy_score(y_data,np.round(y_predict))
print(acc)

'''
출력결과
loss,acc: [0.7876817584037781, 0.75]
0.75
'''
