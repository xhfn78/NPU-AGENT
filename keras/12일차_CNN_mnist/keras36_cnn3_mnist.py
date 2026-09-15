#36-2 카피
import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

#1데이터 
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape,y_train.shape)#(60000, 28, 28,1) (60000,) #흑백데이터라 (60000,28,28)로 표현됨
# print(x_test.shape,y_test.shape) #(10000, 28, 28,1) (10000,)

print(np.max(x_train),np.min(x_train))  #255 0
print(np.max(x_test),np.min(x_test))    #255 0


# ###스케일링 1
# x_train = x_train/255.
# x_test = x_test/255.

# print(np.max(x_train),np.min(x_train))  #1.0 0.0
# print(np.max(x_test),np.min(x_test))    #1.0 0.0


###스케일링 2
x_train = (x_train-127.5)/127.5
x_test = (x_test- 127.5)/127.5

print(np.max(x_train),np.min(x_train))  #-1.0 0.0
print(np.max(x_test),np.min(x_test))    #-1.0 0.0

