# [실습] Dropout 적용 - 산림 수종 (다중 분류)
#
# 층 사이에 Dropout을 넣어서 과적합을 줄여본다.
# Dropout을 넣기 전(keras31_MCP_save_09)과 결과가 어떻게 달라지는지 비교해본다.
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras30/'

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

#2. 모델구성
model = Sequential()
model.add(Dense(200, input_dim=54, activation='relu'))
# Dropout이란?
#   훈련할 때마다 그 층의 뉴런 일부를 무작위로 꺼버린다. Dropout(0.2)면 20%를 끈다.
#   특정 뉴런에만 의존하지 못하게 만들어서 과적합을 줄이는 것이 목적이다.
#
#   중요: 훈련(fit)할 때만 끄고, 평가(evaluate)와 예측(predict)에서는 전부 켠다.
#         그래서 val_loss가 train loss보다 오히려 좋게 나오기도 한다.
#   비율을 너무 크게 잡으면(0.5 이상) 학습 자체가 잘 안 될 수 있다.
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
# ModelCheckpoint(MCP)란?
#   훈련 도중 val_loss가 가장 좋았던 순간의 모델을 파일로 자동 저장해주는 콜백이다.
#
#   EarlyStopping의 restore_best_weights=True 와 뭐가 다른가?
#     EarlyStopping : 최적 가중치를 "메모리 안의 model"에 되돌려준다. 프로그램이 끝나면 사라진다.
#     ModelCheckpoint: 최적 시점의 모델을 "파일"로 남긴다. 나중에 다시 불러 쓸 수 있다.
#
#   주요 옵션
#     monitor='val_loss'    → 무엇을 기준으로 좋고 나쁨을 볼지
#     save_best_only=True   → 좋아졌을 때만 덮어쓴다 (False면 매 epoch 저장해서 파일이 쏟아진다)
#     filepath              → 저장할 경로와 파일명
mcp = ModelCheckpoint(path + 'keras33_dropout09_covtype.keras', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=2000, batch_size=30000, validation_split=0.3, callbacks=[es, mcp])

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