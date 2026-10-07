
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784, 1) / 255.  # Conv1D 변경
x_test = x_test.reshape(-1, 784, 1) / 255.    # Conv1D 변경
#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 3, input_shape=(784, 1), padding='same', activation='relu'))  # Conv1D 변경
model.add(Conv1D(32, 3, padding='same', activation='relu'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(Conv1D(64, 3, padding='same', activation='relu'))  # Conv1D 변경
model.add(Conv1D(64, 3, padding='same', activation='relu'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(Conv1D(128, 3, padding='same', activation='relu'))  # Conv1D 변경
model.add(Conv1D(128, 3, padding='same', activation='relu'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(GlobalAveragePooling1D())  # Conv1D 변경
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2, callbacks=[es, lr], verbose=1)
#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss:', result[0])
print('acc:', result[1])

'''기존 CNN 결과
loss: 0.03946596384048462
acc: 0.9912999868392944
accuracy_score: 0.9913
'''

''' 
# 기존 DNN/CNN 결과
# loss: / acc:
# Conv1D 결과
# loss: / acc:
'''
