import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout

#1.데이터
path = './_data/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
x = train_csv.drop(columns=['casual', 'registered', 'count'])
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=999)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_scaled = scaler.transform(test_csv)

#2.모델구성
# 2-1. 순차형 모델
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
model.summary()

# 2-2. 함수형 모델
input1 = Input(shape=(8,))
dense1 = Dense(10, name='ys1', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(20, name='ys2', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(30, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(20, activation='relu')(drop3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1, activation='relu')(dense5)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

#3.컴파일,훈련
model2.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath='./_save/keras30/keras34_hamsu05_kaggle_bike.keras',
    verbose=1,
)
model2.fit(x_train, y_train, epochs=1000, batch_size=200, validation_split=0.33, verbose=1, callbacks=[es, mcp])

#4.평가,예측
loss = model2.evaluate(x_test, y_test)
y_predict = model2.predict(x_test)
print('loss:', loss)
print('r2:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))

y_submit = model2.predict(test_scaled)
submission['count'] = y_submit
submission.to_csv(path + 'submit/submit_keras34_bike.csv')

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
r2결과값: 0.31977593898773193
RMSE: 146.81258647294175

Dropout 적용 후:
실행 결과의 loss, r2, mse, RMSE 값을 아래에 기록
'''
