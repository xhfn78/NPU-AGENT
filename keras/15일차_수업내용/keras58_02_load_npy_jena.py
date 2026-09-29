# keras58_01_save_npy_jena.py를 먼저 실행한 뒤 npy 불러오기
# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  #메모리 모으기

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import time

#1.데이터
np_path = './_data/kaggle_jena_npy/'
x = np.load(np_path + 'keras58_01_x.npy')
y = np.load(np_path + 'keras58_01_y.npy')
x_predict = np.load(np_path + 'keras58_01_x_predict.npy')
y_cor = np.load(np_path + 'keras58_01_y_cor.npy')

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=333,
    train_size=0.75,
)

# 훈련 데이터로만 scaler를 fit하고 test와 예측 입력은 transform만 적용.
# 3차원 배열을 2차원으로 바꿔 스케일링한 뒤 원래 모양으로 돌림.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1,13)).reshape(-1,144,13)
x_test = scaler.transform(x_test.reshape(-1,13)).reshape(-1,144,13)
x_predict = scaler.transform(x_predict.reshape(-1,13)).reshape(1,144,13)

print('x_train:', x_train.shape, 'y_train:', y_train.shape)
print('x_test:', x_test.shape, 'y_test:', y_test.shape)

#2.모델구성
model = Sequential()
model.add(LSTM(units=64, input_shape=(144,13)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(144))  # y_train 한 개의 정답이 144개이므로 출력도 144개

#3.컴파일,훈련
es = EarlyStopping(
    mode='auto',
    monitor='val_loss',
    patience=40,
    restore_best_weights=True,
    verbose=1,
)

lr = ReduceLROnPlateau(
    mode='auto',
    monitor='val_loss',
    factor=0.5,
    patience=20,
    verbose=1,
)

model.compile(loss='mse', optimizer='adam')  # 풍향 수치 예측은 회귀
start_time = time.time()
model.fit(x_train,y_train,
          epochs=10,
          batch_size=256,
          validation_split=0.2,
          callbacks=[es,lr],
          )
end_time = time.time()

#4.평가,예측
results = model.evaluate(x_test,y_test)
print('loss:', results)
print('걸린시간:', round(end_time-start_time,2), '초')

y_predict = model.predict(x_predict)
print('마지막 144개 실제 풍향:', y_cor)
print('마지막 144개 예측 풍향:', y_predict)
print('마지막 144개 RMSE:', np.sqrt(np.mean((y_cor.reshape(1,144)-y_predict)**2)))
