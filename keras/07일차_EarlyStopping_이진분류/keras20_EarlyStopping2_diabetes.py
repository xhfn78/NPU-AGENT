# [실습] EarlyStopping 적용 - 당뇨병 데이터셋
# epochs를 3000으로 크게 잡아도 EarlyStopping이 과적합 전에 알아서 멈춰준다.
from sklearn.datasets import fetch_california_housing, load_diabetes #캘리포니아 집값 데이터셋,로드 디아벳
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

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

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=10))
model.add(Dense(5))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
from tensorflow.keras.callbacks import EarlyStopping
# EarlyStopping 옵션 설명은 keras20_EarlyStopping1_california.py 에 자세히 적어뒀다.
#   monitor='val_loss'          → 검증 loss를 감시
#   mode='auto'                 → 작아야 좋은지 커야 좋은지 케라스가 알아서 판단
#   patience                    → 개선 없이 몇 epoch까지 참을지 (단위는 epoch)
#   restore_best_weights=True   → 멈춘 시점이 아니라 가장 좋았던 시점의 W, b로 되돌림
es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True
)
hist = model.fit(x_train,y_train, 
                 epochs=3000, 
                 batch_size=10 ,
                 validation_split=0.2,
                 callbacks =[es]
                 )


#4. 평가, 예측
print("=========================================")

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
import matplotlib.pyplot as plt
import platform
# 한글 깨짐 방지. 윈도우는 맑은 고딕, 맥은 AppleGothic을 써야 한다.
plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자 나올때 깨짐방지
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
plt.legend(loc='upper right') #우측상단에 라벨표시

plt.title('당뇨병 Loss') #제목
plt.xlabel('epoch') 
plt.ylabel('loss')
plt.grid()  #격자표시 추가
plt.show()


'''
하이퍼 파라미터 튜닝
#1.데이터 부분
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