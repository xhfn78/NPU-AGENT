import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit

a = np.array(range(1,11))
size = 5   #timestep 사이즈 
print(a.shape)  #(10,)

def split_x(dataset,size):
    aaa = []
    for i in range(len(dataset)- size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x (a , size)

# print(bbb)
# print(bbb.shape)

x = bbb[:,:-1]
y = bbb[ :,-1]

# print(x,y)
# exit()

#2모델구성
model = Sequential()
model.add(LSTM(units=512, input_length=4, input_dim=1))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3컴파일,훈련
# es = EarlyStopping(
#     mode='auto',
#     monitor='val_loss',
#     patience=40,
#     restore_best_weights=True,
#     verbose=1,
# )
# mcp = ModelCheckpoint(
#     mode='auto',
#     monitor='val_loss',
#     patience=40,
#     save_weights_only=True,
#     verbose=1,
# )
# lr = ReduceLROnPlateau(
#     mode='auto',
#     monitor='val_loss',
#     factor=0.5,
#     patience=20,
#     verbose=1,
    
# )

model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs=500,)
# exit()

#4.평가,예측

results = model.evaluate(x,y)

print('loss', results)

x_predict = np.array([7,8,9,10]).reshape(-1,4,1)
y_predict = model.predict(x_predict)

print('[7,8,9,10]의 결과 :', y_predict)

'''

[7,8,9,10]의 결과 : [[10.85953]]


'''
                                          