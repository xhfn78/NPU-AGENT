import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)


#1데이터  
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])

#모델
# model = Perceptron()
# model = LinearSVC()
model = DecisionTreeClassifier()
# model.add(Dense(4, input_dim=2,))
# model.add(Dense(10,))
# model.add(Dense(1,activation='sigmoid'))


#3컴파일,훈련
# model.compile(loss='binary_crossentropy',
#                optimizer = 'adam',
#                metrics =['acc'])
model.fit(x_data,y_data,)


#4평가 예측

y_predict =model.predict(x_data)

loss = model.score(x_data,y_data)
print('loss,acc:',loss)

acc = accuracy_score(y_data,np.round(y_predict))
print(acc)

'''
출력결과
loss,acc: 1.0
1.0
'''
