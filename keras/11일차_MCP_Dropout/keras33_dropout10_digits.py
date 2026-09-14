import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras30/'

#1.데이터
datasets = load_digits()
x = datasets.data
y = pd.get_dummies(datasets.target, dtype=int).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
	x, y, train_size=0.8, random_state=333, stratify=y,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential()
model.add(Dense(30, input_dim=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(path + 'keras33_dropout10_digits.keras', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=500, batch_size=300, validation_split=0.3, callbacks=[es, mcp])

#4.평가,예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', accuracy_score(y_actual, y_predict))


# loss: 0.16165101528167725
# acc: 0.949999988079071
# acc_score: 0.95

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
loss: 0.09308037161827087
acc: 0.97
acc_score: 0.9666666666666667

Dropout 적용 후:
loss: 0.16165101528167725
acc: 0.949999988079071
acc_score: 0.95
'''