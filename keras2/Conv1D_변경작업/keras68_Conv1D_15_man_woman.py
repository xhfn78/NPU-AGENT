
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
np_path = './_data/faces_npy/'
x_train = np.load(np_path + 'faces_01_x_train.npy')
y_train = np.load(np_path + 'faces_01_y_train.npy')
x_test = np.load(np_path + 'faces_01_x_test.npy')
y_test = np.load(np_path + 'faces_01_y_test.npy')
x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2], x_train.shape[3]) / 255.  # Conv1D 변경
x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2], x_test.shape[3]) / 255.    # Conv1D 변경

#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 3, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', padding='same'))  # Conv1D 변경
model.add(Conv1D(32, 3, activation='relu', padding='same'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(Conv1D(64, 3, activation='relu', padding='same'))  # Conv1D 변경
model.add(Conv1D(64, 3, activation='relu', padding='same'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(Conv1D(128, 3, activation='relu', padding='same'))  # Conv1D 변경
model.add(Conv1D(128, 3, activation='relu', padding='same'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(GlobalAveragePooling1D())  # Conv1D 변경
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
model.fit(x_train, y_train, epochs=2000, batch_size=32, validation_split=0.2, callbacks=[es, lr], verbose=1)

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = np.round(model.predict(x_test))
print('loss:', result[0])
print('acc:', result[1])
print('accuracy_score:', accuracy_score(y_test, y_predict))

'''기존 CNN 결과
loss: 0.24654521048069
acc: 0.9017
'''

''' 
# 기존 CNN 결과
# loss: / acc: / accuracy_score:
# Conv1D 결과
# loss: / acc: / accuracy_score:
'''
