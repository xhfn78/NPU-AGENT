import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,Conv2D,Flatten,MaxPool2D


#2.모델구성

model = Sequential()
model.add(Conv2D(10,(2,2), 
                 input_shape=(10,10,1), #9,9,10
                 padding='same', 
                 strides=1,
                 )) 
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=9, 
                 kernel_size=(3,3),  #7,7,9
                 padding='valid',  #디폴트
                 ))

model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 10, 10, 10)        50        padding'same' 적용으로 쉐잎 고정
                                                                 
 conv2d_1 (Conv2D)           (None, 8, 8, 9)           819       padding='valid', 쉐잎 바뀜
                                                                 
=================================================================
Total params: 869
Trainable params: 869
Non-trainable params: 0
_________________________________________________________________
'''
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 5, 5, 10)          50        stride (2) 적용후 바뀐 쉐잎
                                                                 
 conv2d_1 (Conv2D)           (None, 3, 3, 9)           819       
                                                                 
=================================================================
Total params: 869
Trainable params: 869
Non-trainable params: 0
_________________________________________________________________
'''