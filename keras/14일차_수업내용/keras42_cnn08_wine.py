# [실습] Dropout 적용 - 와인 (다중 분류)
#
# 층 사이에 Dropout을 넣어서 과적합을 줄여본다.
# Dropout을 넣기 전(keras31_MCP_save_08)과 결과가 어떻게 달라지는지 비교해본다.
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,GlobalAveragePooling2D,Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_save/keras30/'

#1. 데이터
datasets = load_wine()
x = datasets.data
y = to_categorical(datasets.target)
x_train, x_test, y_train, y_test = train_test_split(
	x, y, train_size=0.8, random_state=333, stratify=y,
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


#2. 모델구성
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
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
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
mcp = ModelCheckpoint(path + 'keras33_dropout08_wine.keras', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', accuracy_score(y_actual, y_predict))

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
loss: 0.13087505102157593
acc_score: 0.9444444444444444

Dropout 적용 후:
실행 결과의 loss, acc, acc_score 값을 아래에 기록
'''