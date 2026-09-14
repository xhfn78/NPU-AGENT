from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터
path = 'c:\\study\\_data\\ddarung\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0).dropna()
x = train_csv.drop(columns='count')
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=666)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.저장 모델 불러오기
files = sorted(Path('./_save/keras30').glob('k31_04_*.keras'), key=lambda file: file.stat().st_mtime)
if not files:
    raise FileNotFoundError('k31_04 체크포인트가 없습니다. keras31_MCP_save_04를 먼저 실행하세요.')
model = load_model(files[-1])

#4.평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss:', loss)
print('r2결과값:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))
