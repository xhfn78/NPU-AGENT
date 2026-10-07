
import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D,MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
scaler = RobustScaler(); x_train = scaler.fit_transform(x_train); x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 13, 1)  # Conv1D 변경
x_test = x_test.reshape(-1, 13, 1)    # Conv1D 변경
#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(13, 1), padding='same', activation='relu'))  # Conv1D 변경
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
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
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

'''기존 CNN 결과
loss: 114.81732177734375
r2: -0.3792889455516546
mse: 114.81732004853671
RMSE: 10.715284412862625
'''

''' 

# Conv1D 결과
loss: 32.004005432128906
r2: 0.6155391056804167
mse: 32.00400444852626
RMSE: 5.65720818500842

'''
