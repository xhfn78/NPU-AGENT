# import ssl
#19-1카피
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import time

#1.데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=666,
)

print(x.shape,y.shape) #(20640, 8) (20640,)

#2.모델구성
model = Sequential()
model.add(Dense(9, input_dim=8))
model.add(Dense(9))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse', optimizer= 'adam')
strat_time = time.time()  #현재 시간을 반환 ,시작시간

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='min',  #헷갈릴떄는 'auto'쓰기 자동으로 min,max 잡아줌
    patience= 15,  #미니멈값 갱신이안되면 10번더 움직임
    restore_best_weights=False,  #10번더 움직인값 말고 최적의 값을 반환해줌 #이거 뻇을떄 좀더 잘나올수도있음 FLASE값넣으면!! 
 )

hist = model.fit(x_train,y_train,
                  epochs=500, 
                  batch_size=32,
                  validation_split=0.2,
                  callbacks=[es], #얼리스탑핑 적용 
                  )
end_time = time.time()  #훈련 끝난 시간을 반환 , 끝시간



#4.평가 ,예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2결과값: ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 

# print('걸린시간 :',round(end_time - strat_time,2),'초')

# print('====================history=======================')
# print(hist) #<keras.src.callbacks.history.History object at 0x000001FABB96A490>
# print('====================hist.history=======================')
# print(hist.history)
# print('====================loss=======================')
# print(hist.history['loss'])
# print('====================val_loss=======================')
# print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')  #맑은 고딕 폰트 적용 한글꺠짐 방지
plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'] ,c='blue', label='val_loss')
plt.legend(loc='upper right') #우측상단에 라벨표시

plt.title('캘리포니아 Loss') #제목
plt.xlabel('epoch') 
plt.ylabel('loss')
plt.grid()  #격자표시 추가
plt.show()



# =========================================
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 682us/step - loss: 0.6196
# loss: 0.6195971369743347
# 645/645 ━━━━━━━━━━━━━━━━━━━━ 0s 350us/step
# 결과값:  [[4.222328 ]
#  [3.9601626]
#  [3.791273 ]
#  ...
#  [0.7560466]
#  [0.8599502]
#  [1.0874628]]