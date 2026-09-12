from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1.데이터

x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[:8]
y_train = y[:8]


x_val = x[8:12]
y_val = y[8:12]

x_test = x[12:]
y_test = y[12:]

# print(x_train.shape,x_val.shape, x_test.shape) #(8,) (4,) (4,)

#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=100 , batch_size=4, 
          verbose=1,
          validation_data = (x_val,y_val)  # 1epochs 당 train >validation 훈련>검증 로직으로 돌아감 
          )
#verbos = 0 :침묵 프로그레스 바 안나오고 결과만 나옴
#verbos = 1 : 디폴트값 (기존에 쓰던방식)
#verbos = 2 : 프로그레스 바 안보임
#verbos = 3 : 프로그레스 바 안보임,에포크 횟수만 나옴 
#verbos = 나머지 : 에포만 나옴. 




#4.평가,예측

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
