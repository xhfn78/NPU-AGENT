import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

path = 'c://study//_data//kaggle_santander//'
save_path = './_save/keras30/'

#1.데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
x = train_csv.drop(['target'], axis=1)
y = to_categorical(train_csv['target'])
x_train, x_test, y_train, y_test = train_test_split(
	x, y, train_size=0.8, random_state=23, stratify=y,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
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

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(save_path + 'keras33_dropout07_santander.keras', monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=40000, validation_split=0.2, callbacks=[es, mcp])

#4.평가,예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', accuracy_score(y_actual, y_predict))

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
loss: 0.2430790215730667
acc: 0.910475
acc_score: 0.910475

Dropout 적용 후:
실행 결과의 loss, acc, acc_score 값을 아래에 기록
'''