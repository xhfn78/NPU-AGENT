import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1.데이터
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=333, stratify=datasets.target,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
# 2-1. 순차형 모델
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 2-2. 함수형 모델
input1 = Input(shape=(30,))
dense1 = Dense(30, name='ys1', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(60, name='ys2', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(70, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(80, activation='relu')(drop3)
dense5 = Dense(60, activation='relu')(dense4)
dense6 = Dense(32, activation='relu')(dense5)
output1 = Dense(1, activation='sigmoid')(dense6)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

#3.컴파일,훈련
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=20, restore_best_weights=True)
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath='./_save/keras30/keras34_hamsu06_cancer.keras',
    verbose=1,
)
model2.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[es, mcp], verbose=1)

#4.평가,예측
result = model2.evaluate(x_test, y_test)
y_predict = np.rint(model2.predict(x_test)).ravel()
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', accuracy_score(y_test, y_predict))

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
loss: 0.04820290952920914
acc_score: 0.9912280701754386

Dropout 적용 후:
실행 결과의 loss, acc, acc_score 값을 아래에 기록
'''
