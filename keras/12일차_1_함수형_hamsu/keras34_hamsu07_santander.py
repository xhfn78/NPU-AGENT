# [실습] 함수형 모델 - 산탄데르 (이진 분류, 캐글)
#
# 같은 모델을 순차형(Sequential)과 함수형(Model) 두 가지로 각각 만들어보고
# summary()로 구조가 동일한지 확인한다.
# 함수형 문법 설명은 keras34_hamsu00.py 참고.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
path = './_data/kaggle_santander/'   #<<< 상대경로 (윈도우/맥 어디서나 동작)
# path = 'c://study//_data//kaggle_santander//'   #<<< 윈도우 절대경로. // 두 개 써도 가능하지만 맥에서는 안 됨
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
x = train_csv.drop(columns='target')
y = to_categorical(train_csv['target'])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=23, stratify=y,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_scaled = scaler.transform(test_csv)

#2. 모델구성
# 2-1. 순차형 모델 (지금까지 쓰던 방식)
model = Sequential()
model.add(Dense(100, input_dim=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()

# 2-2. 함수형 모델
# 함수형은 층을 변수에 담고 괄호로 이어 붙인다.
#   dense1 = Dense(30)(input1)   ← input1을 이 층에 통과시킨다는 뜻
# 위의 순차형과 층 구성이 완전히 같으므로 summary 결과도 같다. 적는 방식만 다르다.
input1 = Input(shape=(200,))
dense1 = Dense(100, name='ys1', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(200, name='ys2', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(300, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(400, activation='relu')(drop3)
dense5 = Dense(200, activation='relu')(dense4)
output1 = Dense(2, activation='softmax')(dense5)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

#3. 컴파일, 훈련
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=100, restore_best_weights=True)
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath='./_save/keras30/keras34_hamsu07_santander.keras',
    verbose=1,
)
model2.fit(x_train, y_train, epochs=100, batch_size=40000, validation_split=0.2, callbacks=[es, mcp], verbose=1)

#4. 평가, 예측
result = model2.evaluate(x_test, y_test)
y_predict = np.argmax(model2.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', accuracy_score(y_actual, y_predict))

y_submit = np.argmax(model2.predict(test_scaled), axis=1)
submission_csv['target'] = y_submit
submission_csv.to_csv(path + 'submit/submit_keras34_santander.csv')

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
loss: 0.2430790215730667
acc: 0.910475
acc_score: 0.910475

Dropout 적용 후:
실행 결과의 loss, acc, acc_score 값을 아래에 기록
'''
