#19-1 카피
# [실습] MinMaxScaler 이해하기 - 캘리포니아 주택 가격
#
# 왜 스케일링을 하는가:
#   뉴런은 h = w1*x1 + w2*x2 + ... 를 계산한다.
#   키(180), 몸무게(60), 연봉(1,000,000,000) 처럼 feature마다 값의 크기가 너무 다르면
#   초기 가중치가 비슷할 때 연봉 항만 압도적으로 커져서 나머지 feature의 영향이 묻힌다.
#   그래서 각 feature의 숫자 범위를 비슷하게 맞춰준다.
#
#   목적은 모든 feature를 똑같이 중요하게 만드는 게 아니라,
#   "단위 차이 때문에" 특정 feature가 과도한 영향을 갖는 문제를 줄이는 것이다.

# import ssl
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것


from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
'''
MinMaxScaler

계산법 :
원값 - min    
----------
max - min  

fit한 데이터의 최솟값이 0, 최댓값이 1이 되도록 변환한다.
예) x의 최솟값이 -1이면 0으로, 최댓값이 10000이면 1로 바뀐다.
'''
from sklearn.preprocessing import MinMaxScaler  #preprocessing(전처리)

scaler = MinMaxScaler()
scaler.fit(x) # x 값을  MinMaxScaler으로 실행시킬 준비
x = scaler.transform(x) # 0~1 값 변환 사이로변환 

# ※ 주의 ※
# 여기서는 나누기 전의 전체 x에 fit을 했다. 이해를 위한 첫 단계라서 이렇게 했지만,
# 이러면 test 데이터의 최소/최대값 정보가 스케일러에 미리 반영되어버린다.
# (아직 안 본 데이터를 훔쳐본 셈이라 평가 점수를 믿을 수 없게 된다)
#
# 제대로 된 순서는 train으로 나눈 뒤 x_train에만 fit하는 것이다.
#   scaler.fit(x_train)
#   x_train = scaler.transform(x_train)
#   x_test  = scaler.transform(x_test)   ← test는 transform만
# 이 올바른 방식은 keras28_Scaler01_california.py 부터 적용한다.
print(x)
print(np.min(x),np.max(x))  #0.0-> min값    1.0000000000000002 -> max값





x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.75,
    random_state=333
)

print(x.shape,y.shape) #(20640, 8) (20640,)

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=8))
model.add(Dense(9))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
start_time = time.time()  #현재 시간을 반환, 시작시간
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True,
)
hist = model.fit(x_train,y_train, epochs=100, batch_size=32  ,validation_split=0.2, callbacks=[es])
end_time = time.time()  #훈련 끝난 시간을 반환, 끝시간



#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2 : ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 

print('걸린시간 :',round(end_time - start_time,2),'초')

# print('====================history=======================')
# print(hist) #<keras.src.callbacks.history.History object at 0x000001FABB96A490>
# print('====================hist.history=======================')
# print(hist.history)
# print('====================loss=======================')
# print(hist.history['loss'])
# print('====================val_loss=======================')
# print(hist.history['val_loss'])

# import matplotlib.pyplot as plt
# import platform
# # 한글 깨짐 방지. 윈도우는 맑은 고딕, 맥은 AppleGothic을 써야 한다.
# plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')
# plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자 나올때 깨짐방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
# plt.legend(loc='upper right') #우측상단에 라벨표시

# plt.title('캘리포니아 Loss') #제목
# plt.xlabel('epoch') 
# plt.ylabel('loss')
# plt.grid()  #격자표시 추가
# plt.show()

'''
cpu-걸린시간
r2 :  0.5980128685579055
mse :  0.5144052108057153
RMSE :  0.7172204757295453
걸린시간 : 38.42 초

gpu-걸린시간 
r2 :  0.5917104903082677
mse :  0.5224700864161658
RMSE :  0.7228209227852814
걸린시간 : 54.75 초


'''