import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#1.데이터
datasets = load_wine()
x = datasets.data
y = to_categorical(datasets.target)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333, stratify=y,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
# 2-1. 순차형 모델
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

# 2-2. 함수형 모델
input1 = Input(shape=(13,))
dense1 = Dense(10, name='ys1', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(20, name='ys2', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(30, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(40, activation='relu')(drop3)
dense5 = Dense(30, activation='relu')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

#3.컴파일,훈련
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=30, restore_best_weights=True)
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath='./_save/keras30/keras34_hamsu08_wine.keras',
    verbose=1,
)
model2.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es, mcp], verbose=1)

#4.평가,예측
result = model2.evaluate(x_test, y_test)
y_predict = np.argmax(model2.predict(x_test), axis=1)
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
