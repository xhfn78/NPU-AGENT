# Jena 데이터를 144개씩 자른 x, y 상태로 npy 저장
import pandas as pd
import numpy as np
import time

#1.데이터
path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(datasets.shape)  # (420551, 14)

y_cor = datasets[-144:]['wd (deg)'].to_numpy(dtype=np.float32)

# 배열로 바꾼 뒤 144개씩 자르기. float32로 저장할 메모리 크기를 줄임.
x_data = datasets[:-288].drop(['wd (deg)'],axis=1).to_numpy(dtype=np.float32)
y_data = datasets[144:-144]['wd (deg)'].to_numpy(dtype=np.float32)

print(x_data.shape)  # (420263, 13)
print(y_data.shape)  # (420263,)

size_x = 144
size_y = 144

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

print('x :', x.shape, 'y :', y.shape)
print('자르는 시간:', round(end_time-start_time,2))

# 마지막 정답 144개 바로 앞의 입력 144개도 스케일링 전 상태로 저장.
x_predict = datasets[-288:-144].drop(['wd (deg)'],axis=1).to_numpy(dtype=np.float32)
x_predict = x_predict.reshape(1,144,13)

print('x_predict:', x_predict.shape, 'y_cor:', y_cor.shape)

#2.npy 저장
np_path = './_data/kaggle_jena_npy/'
np.save(np_path + 'keras58_01_x.npy', arr=x)
np.save(np_path + 'keras58_01_y.npy', arr=y)
np.save(np_path + 'keras58_01_x_predict.npy', arr=x_predict)
np.save(np_path + 'keras58_01_y_cor.npy', arr=y_cor)
print('npy 저장 완료')
