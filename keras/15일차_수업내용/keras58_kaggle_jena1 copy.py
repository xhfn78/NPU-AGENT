# import os
# os.environ['TF_GPU_ALLOCATOR']= 'cuda_malloc_async'  #메모리 모으기 

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import time



path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# print(datasets) #[420551 rows x 14 columns]

y_cor = datasets[-144:]['wd (deg)']  # 정답 데이터 선언
# print(y_cor.shape)

x_data = datasets[:-288].drop(['wd (deg)'],axis=1)
y_data = datasets[144:-144]['wd (deg)']

# print(x_data.shape) #(420263, 13)
# print(y_data.shape) #(420263,)



size_x= 144
size_y= 144
def split_x(dataset, size):
    aaa = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i + size]
        aaa.append(subset)

    return np.array(aaa)

start_time = time.time()
x = split_x(x_data, size_x)
y = split_x(y_data, size_y)
end_time = time.time()

print('x :', x.shape, y.shape)
print('자르는 시간: ',round(end_time-start_time,2))
np_path = './_data/kaggle_jena_npy/'
np.save(np_path + 'keras58_01_x_train.npy', arr = x_data) 
np.save(np_path + 'keras58_01_y_train.npy', arr = y_data) 
exit()
       #
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 random_state=333,
                                                 train_size=0.75
                                                 )



# print(x_train.shape,y_train.shape) #(315090, 144, 13) (315090, 144)
# print(x_test.shape,y_test.shape) #(105030, 144, 13) (105030, 144)

from sklearn.preprocessing import RobustScaler,StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # x_train 기준을 학습(fit)하고 동시에 변환(transform)
x_test = scaler.transform(x_test)         # test는 transform만 (fit 금지)

np_path = './_data/kaggle_jena_npy/'
np.save(np_path + 'keras58_01_x_train.npy', arr = x_data) 
np.save(np_path + 'keras58_01_y_train.npy', arr = y_data) 

exit()
np_path = './_data/kaggle_jena_npy/'
x_train = np.load(np_path + 'keras58_01_x_train.npy', ) 
y_train =np.load(np_path + 'keras58_01_y_train.npy', ) 
x_test = np.load(np_path + 'keras58_01_x_test.npy', ) 
y_test = np.load(np_path + 'keras58_01_y_test.npy',) 
# exit()
#2모델구성
model = Sequential()
model.add(LSTM(units=64, input_shape=(144,13)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3컴파일,훈련
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
model.compile(loss = 'mse',optimizer = 'adam', metrics=['acc'])
model.fit(x_train,y_train,
        epochs=2,
        batch_size=32,
        validation_split=0.2,
        callbacks= [es,lr]
        )

#4.평가,예측

results = model.evaluate(x_test,y_test)

print('loss', results)

x_predict = datasets[288:-144].drop(['wd (deg)'],axis=1)

x_predict = x_predict.to_numpy()

x_predict = x_predict.reshape(1,144,13)
y_predict = model.predict(x_predict)

print('[50,60,70]의 결과 :', y_predict)