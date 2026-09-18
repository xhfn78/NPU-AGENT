# [실습] 함수형 모델 - 손글씨 숫자 (다중 분류)
#
# 같은 모델을 순차형(Sequential)과 함수형(Model) 두 가지로 각각 만들어보고
# summary()로 구조가 동일한지 확인한다.
# 함수형 문법 설명은 keras34_hamsu00.py 참고.
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout,GlobalAveragePooling2D,Conv2D
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_digits()
x = datasets.data
y = pd.get_dummies(datasets.target, dtype=int).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333, stratify=y,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x_train =x_train.reshape(-1,3,3,1)  #(1062, 9) (1062,)
# x_test = x_test.reshape(-1,3,3,1)
print(x_train.shape,y_train.shape) #(1437, 64) (1437, 10)
x_train =x_train.reshape(-1,64,1,1)  #(1062, 9) (1062,)
x_test = x_test.reshape(-1,6,6,2)
print(x_train.shape,y_train.shape) #(331, 10) (331,)
# exit()


#2. 모델구성
# 2-1. 순차형 모델 (지금까지 쓰던 방식)
model = Sequential()
model.add(Conv2D(64,(2,2), input_shape=(64,1,1,), padding='same' ,activation='relu'))
model.add(Conv2D(64,(2,2),  padding='same' ,activation='relu'))
model.add(Conv2D(64 ,(2,2),padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))
model.summary()

# 2-2. 함수형 모델
# 함수형은 층을 변수에 담고 괄호로 이어 붙인다.
#   dense1 = Dense(30)(input1)   ← input1을 이 층에 통과시킨다는 뜻
# 위의 순차형과 층 구성이 완전히 같으므로 summary 결과도 같다. 적는 방식만 다르다.
# input1 = Input(shape=(64,))
# dense1 = Dense(30, name='ys1', activation='relu')(input1)
# drop1 = Dropout(0.2)(dense1)
# dense2 = Dense(50, name='ys2', activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(100, activation='relu')(drop2)
# drop3 = Dropout(0.5)(dense3)
# dense4 = Dense(50, activation='relu')(drop3)
# dense5 = Dense(30, activation='relu')(dense4)
# output1 = Dense(10, activation='softmax')(dense5)
# model2 = Model(inputs=input1, outputs=output1)
# model2.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=200, restore_best_weights=True)
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath='./_save/keras30/keras34_hamsu10_digits.keras',
    verbose=1,
)
model.fit(x_train, y_train, epochs=500, batch_size=300, validation_split=0.3, callbacks=[es, mcp], verbose=1)

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', accuracy_score(y_actual, y_predict))

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
loss: 0.09308037161827087
acc: 0.97
acc_score: 0.9666666666666667

Dropout 적용 후:
실행 결과의 loss, acc, acc_score 값을 아래에 기록
'''
