#36-2 카피
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout,LSTM, Flatten,MaxPool2D,GlobalAveragePooling2D,Input
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
 
#1데이터 
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)#(60000, 28, 28) (60000,) #흑백데이터라 (60000,28,28)로 표현됨
print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)

# print(np.max(x_train),np.min(x_train))  #255 0
# print(np.max(x_test),np.min(x_test))    #255 0
# exit()

###스케일링 1
x_train = x_train/255.
x_test = x_test/255.

# print(np.max(x_train),np.min(x_train))  #1.0 0.0
# print(np.max(x_test),np.min(x_test))    #1.0 0.0

# x_train
# y_train
# ###스케일링 2
# x_train = (x_train-127.5)/127.5
# x_test = (x_test- 127.5)/127.5

# print(np.max(x_train),np.min(x_train))  #1.0 -1.0
# print(np.max(x_test),np.min(x_test))    #1.0 -1.0

model= Sequential()
model.add(LSTM(10,input_shape=(28,28)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))






#3 컴파일,훈련
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001  #디폴트 
learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss ='sparse_categorical_crossentropy',
                optimizer=Adam(learning_rate = learning_rate),
                metrics=['acc'],        
            )  #이진분류에서는 loss = 'binary_crossentropy' 고정
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train,y_train,
          epochs=100,
          batch_size=128,
          verbose=1,
          validation_split =0.2,
          callbacks =[es,],
          )
end_time = time.time()

#평가예측
print(('=====================model.evaluate=================='))
loss = model.evaluate(x_test,y_test, verbose=1)
print('loss: ',loss[0])
print('acc: ',loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)


acc_score = accuracy_score(y_test,y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 :',round(end_time-start_time,2),'초')

'''
cpu걸린시간
loss:  0.04604562371969223
acc:  0.9908000230789185
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
accuracy_score :  0.9908
걸린시간 : 595.72 초

gpu 걸린시간 
loss:  0.04410000890493393
acc:  0.9908999800682068
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9909
걸린시간 : 122.55 초

loss:  0.03946596384048462
acc:  0.9912999868392944
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9913
걸린시간 : 225.34 초

GlobalAveragePooling2D 적용후 
loss:  0.02780984714627266
acc:  0.9922000169754028
accuracy_score :  0.9922
걸린시간 : 209.92 초


loss:  0.021794896572828293
acc:  0.9934999942779541
313/313 [==============================] - 1s 1ms/step
accuracy_score :  0.9935
걸린시간 : 165.61 초


lr 적용후 0.001
===========] - 1s 1ms/step
accuracy_score :  0.9903
걸린시간 : 189.57 초
'''