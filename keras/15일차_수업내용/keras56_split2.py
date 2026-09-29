import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau



a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]
              ]).T

size = 5   #timestep 사이즈 

print(a.shape) # (10, 2)


def split_x(dataset,size):
    aaa = []
    for i in range(len(dataset)- size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x (a , size)
# print(bbb.shape)  #(6, 5, 2)

x = bbb[ :, :-1, :]  #  :, : -1,:
y = bbb[ :,-1, 1]
# y = bbb[ :,-1,-1]

# x = bbb[:,:-1]


# print('==================================')
#print(x)
# print('==================================')
# print(y) #[5 4 3 2 1 0]
# print('==================================')
# print(x.shape,y.shape)  #(6, 4, 2) (6,)




# print(x)
# print(y)
# print(x.shape,y.shape)
# exit()
#2모델구성
model = Sequential()
model.add(LSTM(units=5, input_shape=(4,2),return_sequences=True))
model.add(LSTM(units=10,return_sequences=True))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3.컴파일훈련

model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs=500,)
# exit()

#4.평가,예측

results = model.evaluate(x,y)

print('loss', results)

x_predict = np.array([[7, 3],[8, 2],[9, 1], [10, 0]]).reshape(-1,4,2)
y_predict = model.predict(x_predict)

print('[7, 3],[8, 2],[9, 1], [10, 0]의 결과 :', y_predict)

'''

[7, 3],[8, 2],[9, 1], [10, 0]의 결과 : [[-0.00737311]]


[7, 3],[8, 2],[9, 1], [10, 0]의 결과 : [[0.01500151]]



[7, 3],[8, 2],[9, 1], [10, 0]의 결과 : [[-0.12935157]]


'''