import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

path = './_data/kaggle_bike/'
save_path = './_save/keras30/'

#1.데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=999)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(save_path + 'keras33_dropout05_bike.keras', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=200, validation_split=0.33, callbacks=[es, mcp])

#4.평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss:', loss)
print('r2결과값:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
r2결과값: 0.31977593898773193
RMSE: 146.81258647294175

Dropout 적용 후:
실행 결과의 loss, r2, mse, RMSE 값을 아래에 기록
'''