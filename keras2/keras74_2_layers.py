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
model.add(Dense(10, activation = 'softmax'))

print(len(model.weights)) #30 => 13(vgg16) * 2 (w,b) + 2 + 2 =30
print(len(model.trainable_weights)) #4

print(model.weights)

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

#    layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True