# [실습] 함수형 모델 - 따릉이 대여량 (회귀, 데이콘)
#
# 같은 모델을 순차형(Sequential)과 함수형(Model) 두 가지로 각각 만들어보고
# summary()로 구조가 동일한지 확인한다.
# 함수형 문법 설명은 keras34_hamsu00.py 참고.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout

#1. 데이터
path = './_data/ddarung/'      #<<< 상대경로 (윈도우/맥 어디서나 동작)
# path = 'c:\study\_data\ddarung\'   #<<< 윈도우 절대경로. \ 두 개 써도 가능하지만 맥에서는 안 됨
train_csv = pd.read_csv(path + 'train.csv', index_col=0).dropna()
test_csv = pd.read_csv(path + 'test.csv', index_col=0).fillna(0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)
x = train_csv.drop(columns='count')
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=666)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_scaled = scaler.transform(test_csv)

#2. 모델구성
# 2-1. 순차형 모델 (지금까지 쓰던 방식)
model = Sequential()
model.add(Dense(64, input_dim=9, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.summary()

# 2-2. 함수형 모델
# 함수형은 층을 변수에 담고 괄호로 이어 붙인다.
#   dense1 = Dense(30)(input1)   ← input1을 이 층에 통과시킨다는 뜻
# 위의 순차형과 층 구성이 완전히 같으므로 summary 결과도 같다. 적는 방식만 다르다.
input1 = Input(shape=(9,))
dense1 = Dense(64, name='ys1', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(32, name='ys2', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(16, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(8, activation='relu')(drop3)
dense5 = Dense(4, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

#3. 컴파일, 훈련
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
    filepath='./_save/keras30/keras34_hamsu04_dacon_ddarung.keras',
    verbose=1,
)
model2.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, verbose=1, callbacks=[es, mcp])

#4. 평가, 예측
loss = model2.evaluate(x_test, y_test)
y_predict = model2.predict(x_test)
print('loss:', loss)
print('r2:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))

y_submit = model2.predict(test_scaled)
submission['count'] = y_submit
submission.to_csv(path + 'submit/submit_keras34_ddarung.csv')

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
loss: 2506.461181640625
r2결과값: 0.5851792571021095
RMSE: 50.0645701146075

Dropout 적용 후:
실행 결과의 loss, r2, mse, RMSE 값을 아래에 기록
'''
