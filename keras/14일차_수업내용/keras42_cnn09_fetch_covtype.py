# [실습] 함수형 모델 - 산림 수종 (다중 분류)
#
# 같은 모델을 순차형(Sequential)과 함수형(Model) 두 가지로 각각 만들어보고
# summary()로 구조가 동일한지 확인한다.
# 함수형 문법 설명은 keras34_hamsu00.py 참고.
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout,GlobalAveragePooling2D,Conv2D
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = pd.get_dummies(datasets.target, dtype=int).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=333, stratify=y,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x_train =x_train.reshape(-1,3,3,1)  #(1062, 9) (1062,)
# x_test = x_test.reshape(-1,3,3,1)
print(x_train.shape,y_train.shape) #(331, 10) (331,)
# x_train =x_train.reshape(-1,3,3,1)  #(1062, 9) (1062,)
# x_test = x_test.reshape(-1,3,3,1)
print(x_train.shape,y_train.shape) #(331, 10) (331,)
exit()


model = Sequential()
model.add(Conv2D(8,(2,1), input_shape=(3,3,1,), padding='same' ,activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=50, restore_best_weights=True)
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath='./_save/keras30/keras34_hamsu09_fetch_covtype.keras',
    verbose=1,
)
model.fit(x_train, y_train, epochs=2000, batch_size=30000, validation_split=0.3, callbacks=[es, mcp], verbose=1)

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', accuracy_score(y_actual, y_predict))

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
loss: 0.16669827699661255
acc_score: 0.9417684046263999

Dropout 적용 후:
실행 결과의 loss, acc, acc_score 값을 아래에 기록
'''
