import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import set_random_seed

set_random_seed(333)


#1. 데이터
path = './_data/samsung/'
datasets1 = pd.read_csv(path + '아모레261012.csv',
                       index_col=0, encoding='cp949', thousands=',')
datasets2 = pd.read_csv(path + '하이닉스261012.csv',
                       index_col=0, encoding='cp949', thousands=',')
datasets3 = pd.read_csv(path + '삼성전자261012.csv',
                       index_col=0, encoding='cp949', thousands=',')

# 날짜를 datetime으로 변경하고 날짜 기준 오름차순 정렬
for datasets in [datasets1, datasets2, datasets3]:
    datasets.index = pd.to_datetime(datasets.index, format='%Y/%m/%d')
    datasets.sort_index(inplace=True)

# 데이터 범위 2024-10-12 ~ 2026-10-12
datasets1 = datasets1.loc['2024-10-12':'2026-10-12']
datasets2 = datasets2.loc['2024-10-12':'2026-10-12']
datasets3 = datasets3.loc['2024-10-12':'2026-10-12']

# 아모레, 하이닉스 각각 5개 컬럼 사용
columns = ['시가', '고가', '저가', '종가', '거래량']
datasets1 = datasets1[columns].copy()
datasets2 = datasets2[columns].copy()
datasets3 = datasets3[['시가']].copy()



# 정답: 당일 아모레, 하이닉스로 삼성전자 다음 거래일 시가 예측
# 날짜 오름차순 상태에서 shift(-1)로 다음 거래일 시가를 가져온다.
y_cor = datasets3['시가'].shift(-1)
y_cor.name = '삼성전자_다음시가'

# 행 번호가 아닌 날짜 기준으로 두 종목 결합
x_datasets = pd.concat([datasets1.add_prefix('아모레_'),
                        datasets2.add_prefix('하이닉스_')],
                       axis=1, join='inner')
x_datasets = x_datasets.dropna()

# 입력과 정답의 날짜를 맞추고 정답 없는 마지막 행 제거
datasets = pd.concat([x_datasets, y_cor], axis=1, join='inner')
datasets = datasets.dropna()

# 열 번호 대신 컬럼명 리스트로 각 종목의 입력 선택
x1_columns = ['아모레_시가', '아모레_고가', '아모레_저가', '아모레_종가', '아모레_거래량']
x2_columns = ['하이닉스_시가', '하이닉스_고가', '하이닉스_저가', '하이닉스_종가', '하이닉스_거래량']

x1_data = datasets[x1_columns].to_numpy(dtype=np.float32)
x2_data = datasets[x2_columns].to_numpy(dtype=np.float32)
y_data = datasets['삼성전자_다음시가'].to_numpy(dtype=np.float32)

# 일주일 = 최근 5거래일 (주말, 공휴일 제외)
size_x = 5
def split_x(dataset, size):
    aaa = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i + size]
        aaa.append(subset)

    return np.array(aaa)

x1 = split_x(x1_data, size_x)
x2 = split_x(x2_data, size_x)
# 각 묶음의 마지막 거래일 기준 다음 거래일 시가 하나를 정답으로 사용
# y_data는 위에서 이미 shift(-1)로 다음 거래일 시가에 맞춰 둔 상태
y = y_data[size_x - 1:]

# 내일 예측용 입력: 마지막 거래일을 포함한 최근 5거래일
x1_predict = x_datasets[x1_columns].iloc[-size_x:].to_numpy(dtype=np.float32)
x2_predict = x_datasets[x2_columns].iloc[-size_x:].to_numpy(dtype=np.float32)
x1_predict = split_x(x1_predict, size_x)
x2_predict = split_x(x2_predict, size_x)

print(datasets.head())
print('아모레 :', x1.shape)      # (480, 5, 5)
print('하이닉스 :', x2.shape)    # (480, 5, 5)
print('삼성전자 정답 :', y.shape)  # (480,)
print('학습 입력 기간 :', datasets.index.min(), '~', datasets.index.max())
print('예측 입력 기간 :', x_datasets.index[-size_x], '~', x_datasets.index[-1])
print('예측 입력 :', x1_predict.shape, x2_predict.shape)  # (1, 5, 5), (1, 5, 5)

# 시간 순서대로 분할: 과거 데이터로 학습하고 이후 데이터로 평가
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

# 훈련 데이터로만 fit. 검증, 테스트, 내일 예측 입력은 transform만 적용
# (묶음 수, 5거래일, 5컬럼) -> (전체 거래일, 5컬럼) -> 원래 형태
scaler1 = StandardScaler()
x1_train = scaler1.fit_transform(x1_train.reshape(-1, len(x1_columns))).reshape(-1, size_x, len(x1_columns))
x1_val = scaler1.transform(x1_val.reshape(-1, len(x1_columns))).reshape(-1, size_x, len(x1_columns))
x1_test = scaler1.transform(x1_test.reshape(-1, len(x1_columns))).reshape(-1, size_x, len(x1_columns))
x1_predict = scaler1.transform(x1_predict.reshape(-1, len(x1_columns))).reshape(1, size_x, len(x1_columns))

scaler2 = StandardScaler()
x2_train = scaler2.fit_transform(x2_train.reshape(-1, len(x2_columns))).reshape(-1, size_x, len(x2_columns))
x2_val = scaler2.transform(x2_val.reshape(-1, len(x2_columns))).reshape(-1, size_x, len(x2_columns))
x2_test = scaler2.transform(x2_test.reshape(-1, len(x2_columns))).reshape(-1, size_x, len(x2_columns))
x2_predict = scaler2.transform(x2_predict.reshape(-1, len(x2_columns))).reshape(1, size_x, len(x2_columns))

# y도 훈련 데이터로만 fit하고 검증, 테스트에는 transform만 적용
# 원래 y는 원 단위 평가를 위해 그대로 보관
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

print('train :', x1_train.shape, x2_train.shape, y_train.shape)
print('val :', x1_val.shape, x2_val.shape, y_val.shape)
print('test :', x1_test.shape, x2_test.shape, y_test.shape)

#2-1. 모델1 (아모레, LSTM)
# 입력 형태: (5거래일, 5개 컬럼)
input1 = Input(shape = (size_x, len(x1_columns)))
lstm1 = LSTM(64, name = 'lstm1')(input1)
dense1 = Dense(32, activation = 'relu', name = 'han1')(lstm1)
dense2 = Dense(16, activation = 'relu', name = 'han2')(dense1)
output1 = Dense(5, activation = 'relu', name = 'han3')(dense2)
# model1 = Model(inputs = input1, outputs = output1, name = 'model1')

#2-2. 모델2 (하이닉스, Conv1D)
input21 = Input(shape = (size_x, len(x2_columns)))
                      
conv21 = Conv1D(16, 3, activation = 'relu', padding = 'same', name = 'conv21')(input21)
conv22 = Conv1D(16, 3, activation = 'relu', padding = 'same', name = 'conv22')(conv21)
pool21 = MaxPooling1D(pool_size=2, padding='same', name='pool21')(conv22)
dropout21 = Dropout(0.3, name='dropout21')(pool21)


conv23 = Conv1D(32, 3, activation = 'relu', padding = 'same', name = 'conv23')(dropout21)
conv24 = Conv1D(32, 3, activation = 'relu', padding = 'same', name = 'conv24')(conv23)
pool22 = MaxPooling1D(pool_size=2, padding='same', name='pool22')(conv24)
dropout22 = Dropout(0.3, name='dropout22')(pool22)
gap24 = GlobalAveragePooling1D(name='gap24')(dropout22)

# 다음 Conv1D에는 3차원 dropout 출력을 전달하고, 각 블록의 GAP는 따로 결합
# GAP 출력: 16 + 32 + 64 + 128 = 240개 특성
dense21 = Dense(16, activation = 'relu', name = 'han21')(gap24)
output21 = Dense(5, activation = 'relu', name = 'han22')(dense21)
# model2 = Model(inputs = input21, outputs = output21, name = 'model2')

#2-3. 모델 합치기
merge1 = Concatenate(name = 'mg1')([output1, output21])
merge2 = Dense(10, activation = 'relu', name = 'mg2')(merge1)
merge3 = Dense(5, activation = 'relu', name = 'mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)  # 스케일링된 삼성전자 다음 거래일 시가

model = Model(inputs = [input1, input21], outputs = last_output)
# model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
    verbose=1,
)

lr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.5,
    patience=20,
    verbose=1,
)

start_time = time.time()
hist = model.fit([x1_train, x2_train], y_train_scaled,
                 epochs=1000,
                 batch_size=16,
                 validation_data=([x1_val, x2_val], y_val_scaled),
                 shuffle=False,
                 callbacks=[es, lr],
                 verbose=2,
                 )
end_time = time.time()

# 학습이 끝난 앙상블 모델 가중치 저장
save_path = './save/keras70/'
os.makedirs(save_path, exist_ok=True)
save_name = 'keras70_' + time.strftime('%Y%m%d_%H%M%S') + '.weights.h5'
model.save_weights(save_path + save_name)
print('가중치 저장 완료 :', save_path + save_name)

# 예측할 때 같은 모델 구조와 스케일러를 사용하기 위해 함께 저장
preprocess_name = save_name.replace('.weights.h5', '.preprocess.joblib')
joblib.dump({
    'scaler1': scaler1,
    'scaler2': scaler2,
    'scaler_y': scaler_y,
    'target_scaled': True,
    'size_x': size_x,
    'x1_columns': x1_columns,
    'x2_columns': x2_columns,
    'start_date': '2024-10-12',
    'end_date': '2026-10-12',
    'predict_date': '2026-10-13',
    'model_json': model.to_json(),
}, save_path + preprocess_name)
print('전처리 정보 저장 완료 :', save_path + preprocess_name)

# 예측 파일에서 load_model()로 바로 불러올 수 있도록 전체 모델도 저장
model_name = save_name.replace('.weights.h5', '.model.h5')
model.save(save_path + model_name)
print('모델 저장 완료 :', save_path + model_name)

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test_scaled, )
print('loss, mae (스케일링 기준) :', results)

y_test_predict = model.predict([x1_test, x2_test], )
y_test_predict = scaler_y.inverse_transform(y_test_predict)
y_test_predict = y_test_predict.reshape(-1)

r2 = r2_score(y_test, y_test_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
mae = mean_absolute_error(y_test, y_test_predict)

print('r2 :', r2)
print('RMSE (원) :', rmse)
print('MAE (원) :', mae)
print('학습 epochs :', len(hist.history['loss']))
print('걸린시간 :', round(end_time - start_time, 2), '초')

# 2026-10-12까지 최근 5거래일 데이터로 2026-10-13 삼성전자 시가 예측
y_predict = model.predict([x1_predict, x2_predict], verbose=0)
y_predict = scaler_y.inverse_transform(y_predict)
print('2026-10-13 삼성전자 예측 시가 :', round(float(y_predict[0, 0]), 2), '원')



# 사용한 모델 : ./save/keras70/keras70_20261012_150559.model.h5
# 2026-10-13 삼성전자 예측 시가 : 250031.48 원
