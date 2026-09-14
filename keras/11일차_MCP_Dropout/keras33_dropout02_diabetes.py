from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
import numpy as np

path = './_save/keras30/'

#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=221)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(path + 'keras33_dropout02_diabetes.keras', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=3000, batch_size=10, validation_split=0.2, callbacks=[es, mcp], verbose=1)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss:', loss)
print('r2결과값:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
r2결과값: 0.537927628868109
mse: 2699.0317373389767
RMSE: 51.952206279800826

Dropout 적용 후:
r2결과값: 0.3128578341785333
mse: 4013.69705155258
RMSE: 63.35374536325836
'''
