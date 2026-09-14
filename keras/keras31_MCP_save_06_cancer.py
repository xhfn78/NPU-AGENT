import datetime
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#1.데이터
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=333, stratify=datasets.target)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential([Dense(30, input_dim=30, activation='relu'), Dense(60, activation='relu'), Dense(70, activation='relu'), Dense(80, activation='relu'), Dense(60, activation='relu'), Dense(32, activation='relu'), Dense(1, activation='sigmoid')])

#3.컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
stamp = datetime.datetime.now().strftime('%m%d_%H%M')
filepath = './_save/keras30/k31_06_' + stamp + '-{epoch:04d}-{val_loss:.4f}.keras'
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[es, mcp])

#4.평가,예측
result = model.evaluate(x_test, y_test)
y_predict = np.rint(model.predict(x_test)).ravel()
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', np.mean(y_test == y_predict))
