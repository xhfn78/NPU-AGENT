from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Embedding,GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import accuracy_score


(x_train,y_train), (x_test,y_test) = reuters.load_data(
    num_words=4000, # 단어사전 갯수 ,빈도수 높은 단어순으로 1000개 뽑음.
    # maxlen = 20, # 단어 갯수의 최대길이 제한
    test_split=0.2, 
)
# print(x_train)
# print(x_train.shape ,y_train.shape) #(8982,) (8982,) 전체 train 데이터수
# print(x_test.shape,y_test.shape) #(2246,) (2246,) 전체 train 데이터수
# print(np.unique(y_train)) #class  46
# print(type(x_train)) #<class 'numpy.ndarray'>
# print(type(x_train[0])) #<class 'list'>
# print(len(x_train[0]),len(x_train[1]))
# print('뉴스 기사의 최대길이:', max(len(i) for i in x_train  )) #2376
# print('뉴스 기사의 최소길이:', min(len(i) for i in x_train  )) #13
# print('뉴스 기사의 평균길이:', sum(map(len,x_train))/len(x_train)) #145.5
#전처리 (패드 시퀀스)
############패딩##############

# print(type(x_train))      # numpy.ndarray
# print(type(x_train[0]))   # list
#전처리 (패드 시퀀스)
x_train = pad_sequences(x_train, maxlen=145, padding='pre', truncating='post')
x_test = pad_sequences(x_test, maxlen=145, padding='pre', truncating='post')

# y 원핫
y_train = to_categorical(y_train, num_classes=46)
y_test = to_categorical(y_test, num_classes=46)

# 2. 모델
model = Sequential()
model.add(Embedding(4000,50))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(46, activation='softmax'))

# 3. 컴파일, 훈련
learning_rate = 0.001
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate), 
              metrics=['acc'])
lr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
   
)
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    restore_best_weights=True,
)
model.fit(x_train, y_train, 
        epochs=1000,
        batch_size=32,
        validation_split=0.2, 
        callbacks=[es])

# 4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
# print('acc :', acc)
# 5. 예측
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
# print('y_predict :', y_predict)
# print('y_test :', y_test)
acc_score = accuracy_score(y_test, y_predict)
print('acc_score :', round(acc_score,2))

# 이전 실행 결과
'''
lstm
loss : 1.1795848608016968
acc_score : 0.7203918076580588

gru
1차
acc_score : 0.7444345503116652
2차 
acc_score : 0.7413178984861977



loss : 1.0776792764663696
3차

acc_score : 0.75
'''
