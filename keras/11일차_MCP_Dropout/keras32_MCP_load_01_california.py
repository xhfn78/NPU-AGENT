# [실습] ModelCheckpoint 불러오기 - 캘리포니아 주택 가격 (회귀)
#
# keras31_MCP_save_01 가 저장해둔 체크포인트를 불러와서 평가만 한다.
# 먼저 keras31_MCP_save_01 를 실행해서 저장 파일을 만들어야 동작한다.
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model

#1. 데이터
# 저장할 때와 완전히 똑같이 나누고 똑같이 스케일링해야 한다.
# random_state나 scaler가 다르면 평가 결과를 비교할 수 없다.
datasets = fetch_california_housing()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target,
    train_size=0.75,
    random_state=333,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 저장된 모델 불러오기
# 저장해둔 체크포인트 중 가장 최근 것을 자동으로 찾아서 불러온다.
#   glob으로 해당 패턴의 파일을 모두 찾고, 수정 시각(st_mtime) 순으로 정렬한 뒤 마지막 것을 쓴다.
#   파일명을 직접 적어넣지 않아도 되므로 다시 훈련할 때마다 코드를 고칠 필요가 없다.
# 불러온 모델은 훈련이 끝난 상태이므로 fit 없이 바로 evaluate/predict 한다.
files = sorted(Path('./_save/keras30').glob('k31_01_*.keras'), key=lambda file: file.stat().st_mtime)
if not files:
    raise FileNotFoundError('k31_01 체크포인트가 없습니다. keras31_MCP_save_01를 먼저 실행하세요.')
model = load_model(files[-1])

#4. 평가, 예측
print("=========================================")
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

mse = mean_squared_error(y_test, y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)

# save(keras31_MCP_save_01)에서 훈련 직후 찍은 값과
# load(이 파일)에서 불러와 찍은 값이 정확히 같아야 정상이다.

#save
# loss: 0.5921303033828735
# r2:   0.5372739128508388
# RMSE: 0.7694999511324291

#load
# loss: 0.5921303033828735
# r2:   0.5372739128508388
# RMSE: 0.7694999511324291
