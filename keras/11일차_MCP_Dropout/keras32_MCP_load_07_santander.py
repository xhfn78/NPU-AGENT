# [실습] ModelCheckpoint 불러오기 - 산탄데르 (이진 분류, 캐글)
#
# keras31_MCP_save_07 가 저장해둔 체크포인트를 불러와서 평가만 한다.
# 먼저 keras31_MCP_save_07 를 실행해서 저장 파일을 만들어야 동작한다.
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#1. 데이터
path = './_data/kaggle_santander/'   #<<< 상대경로 (윈도우/맥 어디서나 동작)
# path = 'c://study//_data//kaggle_santander//'   #<<< 윈도우 절대경로. // 두 개 써도 가능하지만 맥에서는 안 됨
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
x = train_csv.drop(columns='target')
y = train_csv['target']
y_onehot = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, train_size=0.8, random_state=23, stratify=y)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 저장된 모델 불러오기
# 저장해둔 체크포인트 중 가장 최근 것을 자동으로 찾아서 불러온다.
#   glob으로 해당 패턴의 파일을 모두 찾고, 수정 시각(st_mtime) 순으로 정렬한 뒤 마지막 것을 쓴다.
#   파일명을 직접 적어넣지 않아도 되므로 다시 훈련할 때마다 코드를 고칠 필요가 없다.
# 불러온 모델은 훈련이 끝난 상태이므로 fit 없이 바로 evaluate/predict 한다.
files = sorted(Path('./_save/keras30').glob('k31_07_*.keras'), key=lambda file: file.stat().st_mtime)
if not files:
    raise FileNotFoundError('k31_07 체크포인트가 없습니다. keras31_MCP_save_07를 먼저 실행하세요.')
model = load_model(files[-1])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', np.mean(y_actual == y_predict))
