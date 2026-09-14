# [실습] validation_split 사용하기
# train_test_split을 두 번 쓰는 대신,
# fit에서 validation_split만 적으면 케라스가 알아서 훈련 데이터 일부를 검증용으로 떼어준다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array(range(1,17))
y = np.array(range(1,17))



x_train,x_test,y_train,y_test = train_test_split(x,y,
    train_size=0.75,                                             
    random_state=333,
    
)




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
        # validation_data = (x_val,y_val)  # 1epochs 당 train >validation 훈련>검증 로직으로 돌아감 
          validation_split=0.33,   # 훈련 데이터 중 33%를 자동으로 검증용으로 할당
          # 주의: 여기서 떼어가는 것은 x_train의 33%지, 전체 데이터의 33%가 아니다.
          # 또 validation_split은 섞지 않고 뒤에서부터 잘라간다.
          )
#verbose = 0 : 침묵. 프로그레스 바 안 나오고 결과만 나옴
#verbose = 1 : 디폴트값 (기존에 쓰던 방식)
#verbose = 2 : 프로그레스 바 안 보임
#verbose = 3 : 프로그레스 바 안 보임, 에포크 횟수만 나옴
#verbose = 나머지 : 에포크만 나옴. 

#4. 평가, 예측

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
