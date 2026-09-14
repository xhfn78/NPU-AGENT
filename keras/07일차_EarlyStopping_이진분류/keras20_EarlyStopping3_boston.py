#11_3 COPY
# [실습] EarlyStopping 적용 - 보스턴 주택 가격 데이터셋
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam') #mse= 원값에서 예측값 뺴고 나온값을 제곱 > 다 더해서 갯수만큼 엔빵
from tensorflow.keras.callbacks import EarlyStopping
# EarlyStopping 옵션 설명은 keras20_EarlyStopping1_california.py 에 자세히 적어뒀다.
#   monitor='val_loss'          → 검증 loss를 감시
#   mode='auto'                 → 작아야 좋은지 커야 좋은지 케라스가 알아서 판단
#   patience                    → 개선 없이 몇 epoch까지 참을지 (단위는 epoch)
#   restore_best_weights=True   → 멈춘 시점이 아니라 가장 좋았던 시점의 W, b로 되돌림
es = EarlyStopping(
    monitor='val_loss',
    mode= 'auto',
    patience= 15,
    restore_best_weights=True,
)
hist = model.fit(x_train,y_train, 
                 epochs=500, 
                 batch_size=16 ,
                 validation_split=0.2,
                 callbacks =[es],
                 )


#4. 평가, 예측
print("=========================================")

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test ,y_predict)
print('r2: ',r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 

# loss: 53.61935043334961
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
# r2:  0.3558761759090454

# loss: 26.69266700744629    로스값이 대략 25 나오는데
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
# r2:  0.6793436562860996
# RMSE :  5.166494759799227  #25가 나온값을 루트 씌워서 제곱 이전으로 돌리면 5
import matplotlib.pyplot as plt
import platform
# 한글 깨짐 방지. 윈도우는 맑은 고딕, 맥은 AppleGothic을 써야 한다.
plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자 나올때 깨짐방지
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
plt.legend(loc='upper right') #우측상단에 라벨표시

plt.title('보스턴 Loss') #제목
plt.xlabel('epoch') 
plt.ylabel('loss')
plt.grid()  #격자표시 추가
plt.show()