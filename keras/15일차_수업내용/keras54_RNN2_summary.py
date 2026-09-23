import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Dropout

#1데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4], 
              [3,4,5],
              [4,5,6],
              [5,6,7],   
              [6,7,8], 
              [7,8,9],
              ])

y= np.array([4,5,6,7,8,9,10])
print(x.shape,y.shape)  #(7, 3) (7,)

x =  x.reshape(x.shape[0],x.shape[1],1)

model =Sequential()
model.add(SimpleRNN(units=3, input_shape = (3,1)))
#3차원으로 들어가서 2(1)차원으로 나옴-> 바로 Dense와 연결가능 
model.add(Dense(7, activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 5)                 35        
                                                                 
 dense (Dense)               (None, 7)                 42        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 85
Trainable params: 85
Non-trainable params: 0 

파라미터 개수 = units*feature+ units*bias + units*units  
_________________________________________________________________
'''
exit()
