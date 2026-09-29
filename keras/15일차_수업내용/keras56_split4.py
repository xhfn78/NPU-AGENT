import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau



a = np.array(range(1,101)).reshape(-1,2) #(50, 2)
# print(a)
x_predict = np.array(range(96,106))
size = 6  #timestep 사이즈 

print(a.shape) # (46, 5, 2)


def split_x(dataset,size):
    aaa = []
    for i in range(len(dataset)- size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x (a , size)
print(bbb.shape) # (46, 5, 2)
# print(bbb) 


x = bbb[:,:-1]
y = bbb[ :, -1,-1 ]

# x = bbb[ :, :-1]  #  :, : -1,:
# y = bbb[ :,-1]
# y = bbb[ :,-1,-1]
# exit()
# x = bbb[:,:-1]
# print(x,y)
print(x.shape,y.shape) #(45, 5, 2) (45,)
# exit()
#2모델구성
model = Sequential()
model.add(LSTM(units=16, input_shape=(5, 2), return_sequences=True))
model.add(LSTM(units=10, return_sequences=True))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3.컴파일훈련

model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs=500,)
# exit()

#4.평가,예측

x_predict = np.array([96, 97, 98, 99, 100], dtype=np.float32)

predictions = []

for i in range(5):
    input_data = x_predict.reshape(1, 5, 1)

    pred = model.predict(input_data, verbose=0)[0][0]

    predictions.append(pred)

    # 가장 오래된 값 제거 + 방금 예측한 값 추가
    x_predict = np.append(x_predict[1:], pred)

print('101~105 예측값:', predictions)



#로스는 0.1이하

'''
loss 0.0035124672576785088
101~106결과 : [[105.37951]]
'''