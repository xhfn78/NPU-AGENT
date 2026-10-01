# keras58_01_save_npy_jena.py를 먼저 실행한 뒤 npy 불러오기
# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  #메모리 모으기

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import time
from tensorflow.keras.optimizers import Adam

#1.데이터
np_path = './_data/kaggle_jena_npy/'
# 앞 파일에서 저장한 훈련용 x, y와 마지막 144개 예측용 데이터를 불러온다.
x = np.load(np_path + 'keras58_01_tdew_x.npy')
y = np.load(np_path + 'keras58_01_tdew_y.npy')
x_predict = np.load(np_path + 'keras58_01_tdew_x_predict.npy')
y_cor = np.load(np_path + 'keras58_01_tdew_y_cor.npy')

# x, y를 같은 순서로 섞어서 훈련 75%, 시험 25%로 나눈다.
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

# print('x_train:', x_train.shape, 'y_train:', y_train.shape)
# print('x_test:', x_test.shape, 'y_test:', y_test.shape)

#2.모델구성
model = Sequential()
# 입력 하나는 144개 시간의 데이터이고, 각 시간에는 wd를 제외한 13개 값이 있다.
model.add(LSTM(units=128, input_shape=(144,13)))
model.add(Dense(144, activation='relu'))
# model.add(Dense(144, activation='relu'))
model.add(Dense(144))  # y_train 한 개의 정답이 144개이므로 출력도 144개
# 학습률을 정해 Adam에 넣는다.
learning_rate = 0.001
#3.컴파일,훈련
# val_loss가 40번 동안 좋아지지 않으면 훈련을 멈추고 가장 좋았던 결과로 돌아간다.
es = EarlyStopping(
    mode='auto',
    monitor='val_loss',
    patience=40,
    restore_best_weights=True,
    verbose=1,
)

# val_loss가 20번 동안 좋아지지 않으면 학습률을 절반으로 줄인다.
lr = ReduceLROnPlateau(
    mode='auto',
    monitor='val_loss',
    factor=0.5,
    patience=20,
    verbose=1,
)

model.compile(loss='mse', 
            optimizer='adam',
            metrics=['acc'],
            )  

start_time = time.time()
model.fit(x_train,y_train,
          epochs=300,
          batch_size=2000,
          validation_split=0.2,
          callbacks=[es,lr],
          verbose=1,
          )
end_time = time.time()

# 훈련이 끝난 모델의 가중치만 저장한다.
save_path = './_save/keras58_02_tdew.weights.h5'
model.save_weights(save_path)
print('가중치 저장 완료:', save_path)

#4.평가,예측
# 훈련에 쓰지 않은 시험 데이터로 loss를 확인한다.
results = model.evaluate(x_test,y_test)
print('loss:', results)
print('걸린시간:', round(end_time-start_time,2), '초')

# 마지막 144개 바로 앞의 데이터를 넣어 wd 값 144개를 예측한다.
y_predict = model.predict(x_predict)

# 예측한 144개와 실제 144개의 차이를 RMSE로 확인한다.
print('마지막 144개 RMSE:', np.sqrt(np.mean((y_cor.reshape(1,144)-y_predict)**2)))

#5.submit 저장
path = './_data/kaggle_jena/'
# 원본 데이터의 마지막 144행을 가져와 wd 열만 예측값으로 바꾼다.
submit = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0).tail(144).copy()
submit['wd (deg)'] = y_predict.reshape(-1)
# 날짜와 다른 열도 함께 CSV에 저장한다.
submit.to_csv(np_path + 'keras58_jena_submit.csv')
print('submit 저장 완료:', np_path + 'keras58_jena_submit.csv')


'''
- acc: 0.0840
loss: [0.9203147292137146, 0.08396648615598679]
걸린시간: 1759.81 초
1/1 [==============================] - 0s 255ms/step
마지막 144개 RMSE: 4.394194
submit 저장 완료: ./_data/kaggle_jena_npy/keras58_jena_submit.csv
'''