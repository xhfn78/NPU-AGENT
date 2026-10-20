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

model = VGG16(
    weights = 'imagenet',  
    include_top = False,   # False로 하면 input_shape변경 가능, Flatten과 하단의 FC layer 사라짐
    input_shape = (32, 32, 3),
)

model.trainable = True
# vgg16.summary()


model.layers[11].trainable = False
model.summary()
# exit()
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

#       layer Type    Layer Name  Layer Trainable
# 0     InputLayer       input_1             True
# 1         Conv2D  block1_conv1             True
# 2         Conv2D  block1_conv2             True
# 3   MaxPooling2D   block1_pool             True
# 4         Conv2D  block2_conv1             True
# 5         Conv2D  block2_conv2             True
# 6   MaxPooling2D   block2_pool             True
# 7         Conv2D  block3_conv1             True
# 8         Conv2D  block3_conv2             True
# 9         Conv2D  block3_conv3             True
# 10  MaxPooling2D   block3_pool             True
# 11        Conv2D  block4_conv1            False
# 12        Conv2D  block4_conv2             True
# 13        Conv2D  block4_conv3             True
# 14  MaxPooling2D   block4_pool             True
# 15        Conv2D  block5_conv1             True
# 16        Conv2D  block5_conv2             True
# 17        Conv2D  block5_conv3             True
# 18  MaxPooling2D   block5_pool             True