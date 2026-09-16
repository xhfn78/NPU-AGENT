from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D

model = Sequential()
model.add(Conv2D(10,(2,2),input_shape=(5,5,1),))
model.add(Conv2D(5, (2,2)))
model.add(Dense(10))

model.summary()
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50        
                                                                 
#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205       
# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0
# _________________________________________________________________

