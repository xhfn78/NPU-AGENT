import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


path = './_data/samsung/'
save_path = './save/keras70/'
model_path = save_path + 'keras70_20261012_150559.model.h5'

#1. 데이터 불러오기
# 모델 파일에는 스케일러가 없으므로 학습 당시와 같은 CSV, 기간, 분할 비율 사용
datasets1 = pd.read_csv(path + '아모레261012.csv',
                       index_col=0, encoding='cp949', thousands=',')
datasets2 = pd.read_csv(path + '하이닉스261012.csv',
                       index_col=0, encoding='cp949', thousands=',')
datasets3 = pd.read_csv(path + '삼성전자261012.csv',
                       index_col=0, encoding='cp949', thousands=',')

for datasets in [datasets1, datasets2, datasets3]:
    datasets.index = pd.to_datetime(datasets.index, format='%Y/%m/%d')
    datasets.sort_index(inplace=True)

datasets1 = datasets1.loc['2024-10-12':'2026-10-12']
datasets2 = datasets2.loc['2024-10-12':'2026-10-12']
datasets3 = datasets3.loc['2024-10-12':'2026-10-12']

columns = ['시가', '고가', '저가', '종가', '거래량']
datasets1 = datasets1[columns].copy()
datasets2 = datasets2[columns].copy()
datasets3 = datasets3[['시가']].copy()

x1_columns = ['아모레_시가', '아모레_고가', '아모레_저가', '아모레_종가', '아모레_거래량']
x2_columns = ['하이닉스_시가', '하이닉스_고가', '하이닉스_저가', '하이닉스_종가', '하이닉스_거래량']

y_cor = datasets3['시가'].shift(-1)
y_cor.name = '삼성전자_다음시가'
x_datasets = pd.concat([datasets1.add_prefix('아모레_'),
                        datasets2.add_prefix('하이닉스_')],
                       axis=1, join='inner').dropna()
datasets = pd.concat([x_datasets, y_cor], axis=1, join='inner').dropna()

size_x = 5
def split_x(dataset, size):
    aaa = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i + size]
        aaa.append(subset)

    return np.array(aaa)

x1_data = datasets[x1_columns].to_numpy(dtype=np.float32)
x2_data = datasets[x2_columns].to_numpy(dtype=np.float32)
y_data = datasets['삼성전자_다음시가'].to_numpy(dtype=np.float32)
x1 = split_x(x1_data, size_x)
x2 = split_x(x2_data, size_x)
y = y_data[size_x - 1:]

# 학습 당시 스케일러 기준을 재현하기 위한 분할. 모델 학습은 하지 않음
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y,
    train_size=0.8,
    shuffle=False,
)
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
    x1_train, x2_train, y_train,
    train_size=0.9,
    shuffle=False,
)

scaler1 = StandardScaler()
scaler1.fit(x1_train.reshape(-1, len(x1_columns)))
scaler2 = StandardScaler()
scaler2.fit(x2_train.reshape(-1, len(x2_columns)))
scaler_y = StandardScaler()
scaler_y.fit(y_train.reshape(-1, 1))

# 예측 입력: 2026-10-12까지 최근 5거래일
x1_predict = x_datasets[x1_columns].iloc[-size_x:].to_numpy(dtype=np.float32)
x2_predict = x_datasets[x2_columns].iloc[-size_x:].to_numpy(dtype=np.float32)
x1_predict = scaler1.transform(x1_predict).reshape(1, size_x, len(x1_columns))
x2_predict = scaler2.transform(x2_predict).reshape(1, size_x, len(x2_columns))

#2. 구조와 가중치가 함께 저장된 모델 불러오기
model = load_model(model_path, compile=False)

#3. 예측값 확인 (compile, model.fit 없이 실행)
y_predict = model.predict([x1_predict, x2_predict],)
y_predict = scaler_y.inverse_transform(y_predict)

print('사용한 모델 :', model_path)
print('2026-10-13 삼성전자 예측 시가 :', round(float(y_predict[0, 0]), 2), '원')
