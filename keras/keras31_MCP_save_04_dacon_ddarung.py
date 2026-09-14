import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터
path = 'c:\\study\\_data\\ddarung\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0).dropna()
x = train_csv.drop(columns='count')
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=666)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential([Dense(64, input_dim=9, activation='relu'), Dense(32, activation='relu'), Dense(16, activation='relu'), Dense(8, activation='relu'), Dense(4, activation='relu'), Dense(1)])

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
stamp = datetime.datetime.now().strftime('%m%d_%H%M')
filepath = './_save/keras30/k31_04_' + stamp + '-{epoch:04d}-{val_loss:.4f}.keras'
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[es, mcp])

#4.평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss:', loss)
print('r2결과값:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))
