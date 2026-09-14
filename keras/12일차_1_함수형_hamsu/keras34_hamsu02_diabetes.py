# [실습] 함수형 모델 - 당뇨병 (회귀)
#
# 같은 모델을 순차형(Sequential)과 함수형(Model) 두 가지로 각각 만들어보고
# summary()로 구조가 동일한지 확인한다.
# 함수형 문법 설명은 keras34_hamsu00.py 참고.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.75, random_state=221,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# 2-1. 순차형 모델 (지금까지 쓰던 방식)
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
model.summary()

# 2-2. 함수형 모델
# 함수형은 층을 변수에 담고 괄호로 이어 붙인다.
#   dense1 = Dense(30)(input1)   ← input1을 이 층에 통과시킨다는 뜻
# 위의 순차형과 층 구성이 완전히 같으므로 summary 결과도 같다. 적는 방식만 다르다.
input1 = Input(shape=(10,))
dense1 = Dense(3, name='ys1', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(10, name='ys2', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(15, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(20, activation='relu')(drop3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

input1 = Input(shape=(10,))
dense1 = Dense(3, name='ys1', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(10, name = 'ys2', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(15, name='ys3',activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(20, name='ys3',activation='relu')(drop3)
dense5 = Dense(10, name='ys3',activation='relu')(dense4)
model2 =Model(inputs= input1, outpus=output1)
#3. 컴파일, 훈련
model2.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='auto', patience=15, restore_best_weights=True)
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath='./_save/keras30/keras34_hamsu02_diabetes.keras',
    verbose=1,
)
model2.fit(x_train, y_train, epochs=3000, batch_size=10, validation_split=0.2, callbacks=[es, mcp], verbose=1)

#4. 평가, 예측
loss = model2.evaluate(x_test, y_test)
y_predict = model2.predict(x_test)
print('loss:', loss)
print('r2:', r2_score(y_test, y_predict))
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
