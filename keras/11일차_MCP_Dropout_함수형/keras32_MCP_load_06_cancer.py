from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#1.데이터
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=333, stratify=datasets.target)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.저장 모델 불러오기
files = sorted(Path('./_save/keras30').glob('k31_06_*.keras'), key=lambda file: file.stat().st_mtime)
if not files:
    raise FileNotFoundError('k31_06 체크포인트가 없습니다. keras31_MCP_save_06를 먼저 실행하세요.')
model = load_model(files[-1])

#4.평가,예측
result = model.evaluate(x_test, y_test)
y_predict = np.rint(model.predict(x_test)).ravel()
print('loss:', result[0])
print('acc:', result[1])
print('acc_score:', np.mean(y_test == y_predict))
