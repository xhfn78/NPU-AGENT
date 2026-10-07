
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
datasets = load_diabetes(); x = datasets.data; y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=221)
scaler = RobustScaler(); x_train = scaler.fit_transform(x_train); x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 10, 1)  # Conv1D 변경
x_test = x_test.reshape(-1, 10, 1)    # Conv1D 변경
#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(10, 1), padding='same', activation='relu'))  # Conv1D 변경
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
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
model.fit(x_train, y_train, epochs=500, batch_size=16, validation_split=0.2, callbacks=[es, lr], verbose=1)
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss:', loss)
print('r2:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))

''' 
# 기존 DNN/CNN 결과
# loss: / r2: / mse: / RMSE:0
loss: 0.3257753252983093
r2: 0.7454196921017856
mse: 0.32577519703570174
RMSE: 0.570767200385325


# Conv1D 결과
loss: 4373.576171875
r2: 0.25124671250552255
mse: 4373.576549132504
RMSE: 66.13302162409111

'''



