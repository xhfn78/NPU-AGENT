import datetime
import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#1.데이터
datasets = load_wine()
x = datasets.data
y = to_categorical(datasets.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333, stratify=y)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential([Dense(10, input_dim=13, activation='relu'), Dense(20, activation='relu'), Dense(30, activation='relu'), Dense(40, activation='relu'), Dense(30, activation='relu'), Dense(3, activation='softmax')])

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
stamp = datetime.datetime.now().strftime('%m%d_%H%M')
filepath = './_save/keras30/k31_08_' + stamp + '-{epoch:04d}-{val_loss:.4f}.keras'
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es, mcp])

#4.평가,예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', np.mean(y_actual == y_predict))
