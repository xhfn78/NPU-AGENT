# [실습] ModelCheckpoint 저장 - 당뇨병 (회귀)
#
# 훈련하면서 val_loss가 가장 좋았던 시점의 모델을
# ./_save/keras30/ 아래에 k31_02_시각-epoch-val_loss.keras 형태로 저장한다.
# 짝이 되는 불러오기 파일은 keras32_MCP_load_02 이다.
from sklearn.datasets import fetch_california_housing, load_diabetes #캘리포니아 집값 데이터셋,로드 디아벳
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

path = './_save/keras30/' 


#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape,y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=221,
    train_size=0.75, 

)

from sklearn.preprocessing import RobustScaler

##############################################################################
scaler = RobustScaler()
##############################################################################

scaler.fit(x_train) # x 값을  MinMaxScaler으로 실행시킬 준비
x_train = scaler.fit_transform(x_train) # 0~1 값 변환 사이로변환
x_test = scaler.transform(x_test) 

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,))



#3. 컴파일, 훈련
import datetime
date = datetime.datetime.now() 
print(date) #2026-09-14 11:41:17.149590
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')
print(date)
print(type(date))
path = './_save/keras30/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([path,'k30_',date,'-',filename])


model.compile(loss='mse', optimizer= 'adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True
)
# ModelCheckpoint(MCP)란?
#   훈련 도중 val_loss가 가장 좋았던 순간의 모델을 파일로 자동 저장해주는 콜백이다.
#
#   EarlyStopping의 restore_best_weights=True 와 뭐가 다른가?
#     EarlyStopping : 최적 가중치를 "메모리 안의 model"에 되돌려준다. 프로그램이 끝나면 사라진다.
#     ModelCheckpoint: 최적 시점의 모델을 "파일"로 남긴다. 나중에 다시 불러 쓸 수 있다.
#
#   주요 옵션
#     monitor='val_loss'    → 무엇을 기준으로 좋고 나쁨을 볼지
#     save_best_only=True   → 좋아졌을 때만 덮어쓴다 (False면 매 epoch 저장해서 파일이 쏟아진다)
#     filepath              → 저장할 경로와 파일명
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode= 'auto',
    save_best_only=True,
    filepath = filepath,
    verbose=1,
)


hist = model.fit(x_train,y_train, 
                 epochs=3000, 
                 batch_size=10 ,
                 validation_split=0.2,
                 callbacks =[es,mcp]
                 )


#4. 평가, 예측
print("=========================================")

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
# results = model.predict(x)
# print('결과값: ' ,results)
#랜덤 442
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2857.6667 
# loss: 2857.666748046875
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 결과값:  [[210.36603 ]

#랜덤 221
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2459.2292 
# loss: 2459.229248046875
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 

#랜덤 3333
# 42/42 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2882.3848 
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 3071.2979 
# loss: 3071.2978515625
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 결과값:  [[204.20064 ]
# import matplotlib.pyplot as plt
# import platform
# 한글 깨짐 방지. 윈도우는 맑은 고딕, 맥은 AppleGothic을 써야 한다.
plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')
# plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
# plt.legend(loc='upper right') #우측상단에 라벨표시

# plt.title('당뇨병 Loss') #제목
# plt.xlabel('epoch') 
# plt.ylabel('loss')
# plt.grid()  #격자표시 추가
# plt.show()


'''
하이퍼 파라미터 튜닝
#1. 데이터부분
random_state
train_size
#2.
레이어의 깊이
노드의갯수
#3
epoch
batch_size
'''


"""
1차시도
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2637.873291015625
r2결과값:  0.5483979249211794
mse :  2637.8732195134085
RMSE :  51.36022994023107
"""

"""
2차시도 --- MINMAX-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2574.98388671875
r2결과값:  0.5591645248939214
mse :  2574.983947518029
RMSE :  50.744299655409854
"""



"""
3차시도 --- standard-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2451.989013671875
r2결과값:  0.5802211882185655
mse :  2451.9889230450026
RMSE :  49.5175617639338
"""

"""
3차시도 --- standard-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2661.037353515625
r2결과값:  0.5444322418537058
mse :  2661.0373494810156
RMSE :  51.58524352449076
"""



# r2결과값:  0.5541249992369102
# mse :  2604.4205477979554
# RMSE :  51.03352376426652