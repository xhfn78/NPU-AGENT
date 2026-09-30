# Jena 데이터를 144개씩 자른 x, y 상태로 npy 저장
# 이 파일을 먼저 실행하고 keras58_02_load_npy_jena.py에서 저장한 파일을 불러온다.
import pandas as pd
import numpy as np
import time

#1.데이터
path = './_data/kaggle_jena/'
# 첫 번째 Date Time 열을 행 이름으로 두고 나머지 14개 열을 읽는다.
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(datasets.shape)  # (420551, 14)

# 마지막 144개의 실제 Tdew 값은 훈련에 쓰지 않고 예측 결과와 비교할 정답으로 둔다.
y_cor = datasets[-144:]['Tdew (degC)'].to_numpy(dtype=np.float16)

# 배열로 바꾼 뒤 144개씩 자르기. float32로 저장할 메모리 크기를 줄임.
# x는 Tdew 열을 빼고 마지막 288개를 남긴다. y는 144개 뒤부터 시작해 마지막 144개를 남긴다.
# 따라서 x의 144개를 보고 그다음 144개의 Tdew 값을 맞히게 된다.
x_data = datasets[:-288].drop(['Tdew (degC)'],axis=1).to_numpy(dtype=np.float16)
y_data = datasets[144:-144]['Tdew (degC)'].to_numpy(dtype=np.float16)

print(x_data.shape)  # (420263, 13)
print(y_data.shape)  # (420263,)

# 입력 144개와 정답 144개를 한 묶음으로 자른다.
size_x = 144
size_y = 144

def split_x(dataset, size):
    aaa = []
    # 한 칸씩 옮기면서 길이가 size인 데이터를 만든다.
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i + size]
        aaa.append(subset)
    return np.array(aaa)

# x, y를 각각 144개씩 자르고 걸린 시간을 확인한다.
start_time = time.time()
x = split_x(x_data, size_x)
y = split_x(y_data, size_y)
end_time = time.time()

print('x :', x.shape, 'y :', y.shape)
print('자르는 시간:', round(end_time-start_time,2))

# 마지막 정답 144개 바로 앞의 입력 144개도 스케일링 전 상태로 저장.
x_predict = datasets[-288:-144].drop(['Tdew (degC)'],axis=1).to_numpy(dtype=np.float16)
# 모델에 넣을 입력 하나이므로 (1, 144, 13)으로 바꾼다.
x_predict = x_predict.reshape(1,144,13)

print('x_predict:', x_predict.shape, 'y_cor:', y_cor.shape)

#2.npy 저장
np_path = './_data/kaggle_jena_npy/'
# 훈련용 x, y와 마지막 144개 예측에 쓸 x_predict, 비교할 y_cor를 저장한다.
np.save(np_path + 'keras58_01_tdew_x.npy', arr=x)
np.save(np_path + 'keras58_01_tdew_y.npy', arr=y)
np.save(np_path + 'keras58_01_tdew_x_predict.npy', arr=x_predict)
np.save(np_path + 'keras58_01_tdew_y_cor.npy', arr=y_cor)
print('npy 저장 완료')
