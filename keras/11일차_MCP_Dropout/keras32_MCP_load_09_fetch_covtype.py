# [실습] ModelCheckpoint 불러오기 - 산림 수종 (다중 분류)
#
# keras31_MCP_save_09 가 저장해둔 체크포인트를 불러와서 평가만 한다.
# 먼저 keras31_MCP_save_09 를 실행해서 저장 파일을 만들어야 동작한다.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = pd.get_dummies(datasets.target, dtype=int).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=333, stratify=y)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 저장된 모델 불러오기
# 저장해둔 체크포인트 중 가장 최근 것을 자동으로 찾아서 불러온다.
#   glob으로 해당 패턴의 파일을 모두 찾고, 수정 시각(st_mtime) 순으로 정렬한 뒤 마지막 것을 쓴다.
#   파일명을 직접 적어넣지 않아도 되므로 다시 훈련할 때마다 코드를 고칠 필요가 없다.
# 불러온 모델은 훈련이 끝난 상태이므로 fit 없이 바로 evaluate/predict 한다.
files = sorted(Path('./_save/keras30').glob('k31_09_*.keras'), key=lambda file: file.stat().st_mtime)
if not files:
    raise FileNotFoundError('k31_09 체크포인트가 없습니다. keras31_MCP_save_09를 먼저 실행하세요.')
model = load_model(files[-1])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', np.mean(y_actual == y_predict))
