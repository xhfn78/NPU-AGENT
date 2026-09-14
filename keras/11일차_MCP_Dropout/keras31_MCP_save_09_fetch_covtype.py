import datetime
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = pd.get_dummies(datasets.target, dtype=int).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=333, stratify=y)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential([Dense(200, input_dim=54, activation='relu'), Dense(300, activation='relu'), Dense(300, activation='relu'), Dense(200, activation='relu'), Dense(100, activation='relu'), Dense(7, activation='softmax')])

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
stamp = datetime.datetime.now().strftime('%m%d_%H%M')
filepath = './_save/keras30/k31_09_' + stamp + '-{epoch:04d}-{val_loss:.4f}.keras'
es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=2000, batch_size=30000, validation_split=0.3, callbacks=[es, mcp])

#4.평가,예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', np.mean(y_actual == y_predict))
