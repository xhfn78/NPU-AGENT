from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#1.데이터
path = 'c://study//_data//kaggle_santander//'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
x = train_csv.drop(columns='target')
y = train_csv['target']
y_onehot = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, train_size=0.8, random_state=23, stratify=y)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.저장 모델 불러오기
files = sorted(Path('./_save/keras30').glob('k31_07_*.keras'), key=lambda file: file.stat().st_mtime)
if not files:
    raise FileNotFoundError('k31_07 체크포인트가 없습니다. keras31_MCP_save_07를 먼저 실행하세요.')
model = load_model(files[-1])

#4.평가,예측
result = model.evaluate(x_test, y_test)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', np.mean(y_actual == y_predict))
