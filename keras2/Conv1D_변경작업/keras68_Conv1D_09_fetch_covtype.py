
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
datasets = fetch_covtype(); x = datasets.data; y = to_categorical(datasets.target - 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333, stratify=y)
scaler = RobustScaler(); x_train = scaler.fit_transform(x_train); x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 54, 1)  # Conv1D 변경
x_test = x_test.reshape(-1, 54, 1)    # Conv1D 변경
#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(54, 1), padding='same', activation='relu'))  # Conv1D 변경
model.add(Conv1D(32, 2, padding='same', activation='relu'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(Conv1D(64, 2, padding='same', activation='relu'))  # Conv1D 변경
model.add(Conv1D(64, 2, padding='same', activation='relu'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(Conv1D(128, 2, padding='same', activation='relu'))  # Conv1D 변경
model.add(Conv1D(128, 2, padding='same', activation='relu'))  # Conv1D 변경
model.add(Dropout(0.2))
model.add(MaxPool1D(2))  # Conv1D 변경
model.add(GlobalAveragePooling1D())  # Conv1D 변경
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es, lr], verbose=1)
#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', accuracy_score(y_actual, y_predict))

'''기존 CNN 결과
loss: 0.16669827699661255
acc_score: 0.9417684046263999
'''

''' 
# 기존 DNN/CNN 결과
# loss: / acc: / acc_score:
# Conv1D 결과
# loss: / acc: / acc_score:
'''
