import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


from keras.applications import VGG16

#2. 모델

vgg16 = VGG16(
    weights = 'imagenet',  
    include_top = False,   # False로 하면 input_shape변경 가능, Flatten과 하단의 FC layer 사라짐
    input_shape = (32, 32, 3),
)

vgg16.trainable = False
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(10, activation = 'softmax'))

model.layers[3].trainable = False
model.summary()

import pandas as pd
pd.set_option('max_colwidth',None)
layers=[
    (type(layer).__name__,layer.name,layer.trainable) 
    for layer in model.layers
]

results = pd.DataFrame(
    layers,
    columns=['layer Type','Layer Name','Layer Trainable'],)
print(results)

# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True
# 4       Dense    dense_2             True
# 5       Dense    dense_3             True
# 6       Dense    dense_4             True