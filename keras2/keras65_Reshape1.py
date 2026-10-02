#36-2 카피
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn.metrics import accuracy_score
 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape) #(60000, 28, 28)

###스케일링 1
x_train = x_train/255.
x_test = x_test/255.



from tensorflow.keras.layers import Reshape


#2. 모델구성
model = Sequential()
model.add(Dense(28*10,input_shape=(28,28))) #   (None, 28, 280)  ->(n,28,28*10) 
model.add(Reshape(target_shape=(28,28,10)))
model.add(Conv2D(64,(3,3),input_shape =(28,28,10)))  
model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu',padding='same')) 
model.add(Conv2D(16,(2,2),activation='relu')) #(None, 22, 22, 16)
model.add(Conv2D(16,(2,2),activation='relu')) #(None, 20, 20, 16) 4차원데이터에서 마지막 (None, 10) 로 붙이기 위해서 flatten 사용
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu')) #아웃풋 node의 갯수= units

model.add(Dense(10, activation='softmax'))#(None, 10) 
# model.summary()


#3 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam',
              metrics= ['acc']
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train,y_train,
          epochs=1000,
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

y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

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



데이터 증폭후 
loss:  0.028132364153862
acc:  0.9908999800682068
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.9909
걸린시간 : 165.79 초

'''