# [실습] train_test_split을 두 번 써서 train / validation / test 만들기
# 슬라이싱은 앞뒤로 잘리는 문제가 있으니 랜덤 분할을 두 번 한다.
#   1단계: 전체 → train + test
#   2단계: 그 train을 다시 → train + validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array(range(1,17))
y = np.array(range(1,17))



x_train,x_test,y_train,y_test = train_test_split(x,y,
    random_state=333,
    
)

# 주의: 두 번째는 x가 아니라 x_train을 다시 쪼갠다.
# test는 이미 빼놨으므로 건드리면 안 된다.
x_train,x_val,y_train,y_val =train_test_split(x_train,y_train,
    random_state=333,
)

# print(x_train)
# print(x_val)
# print(x_test)

# exit()

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=100 , batch_size=4, 
          verbose=1,
          validation_data = (x_val,y_val)  # 1epochs 당 train >validation 훈련>검증 로직으로 돌아감 
          # validation(검증)은 훈련 중간에 "지금 잘 되고 있나"를 확인하는 용도다.
          # train으로 w를 갱신하고, val로는 갱신하지 않고 점수만 매긴다.
          # test는 맨 마지막에 딱 한 번 쓰는 최종 시험지라서 훈련 중에는 건드리지 않는다.
          )
#verbose = 0 : 침묵. 프로그레스 바 안 나오고 결과만 나옴
#verbose = 1 : 디폴트값 (기존에 쓰던 방식)
#verbose = 2 : 프로그레스 바 안 보임
#verbose = 3 : 프로그레스 바 안 보임, 에포크 횟수만 나옴
#verbose = 나머지 : 에포크만 나옴. 




#4. 평가, 예측

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
