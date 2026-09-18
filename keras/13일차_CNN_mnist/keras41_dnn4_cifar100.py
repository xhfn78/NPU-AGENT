import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn.metrics import accuracy_score
 
#1데이터 
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape,y_train.shape)#(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# print(np.max(x_train),np.min(x_train))  #255 0
# print(np.max(x_test),np.min(x_test))    #255 0

# print(np.unique(y,return_counts=True))
# exit()
###스케일링 1
x_train = x_train/255.
x_test = x_test/255.

# print(np.max(x_train),np.min(x_train))  #1.0 0.0
# print(np.max(x_test),np.min(x_test))    #1.0 0.0
# exit()

# ###스케일링 2
# x_train = (x_train-127.5)/127.5
# x_test = (x_test- 127.5)/127.5

x_train =x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)
# print(np.max(x_train),np.min(x_train))  #1.0 -1.0
# print(np.max(x_test),np.min(x_test))    #1.0 -1.0

#######################원한 인코더###########################
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)  
y_test = y_test.reshape(-1,1) 
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
# print(y_train.shape,y_test.shape)  #(60000, 100) (10000, 100)

#2. 모델구성 — CIFAR-100용 VGG16 변형
# VGG16: Conv 13개 + Dense 3개 = 가중치를 학습하는 16개 층.
# 3×3 Conv를 2·2·3·3·3개씩 쌓고, 각 블록 끝에서 MaxPooling으로 크기를 절반으로 줄입니다.
# 원본의 입력 224×224와 Dense 4096→4096→1000을 32×32와 512→256→100으로 조정했습니다.
# 사전학습 가중치 없이 직접 구성하여 처음부터 학습합니다.
# Conv 공통: padding='same'으로 가로·세로 유지, ReLU로 비선형성 추가.
model = Sequential()

# Block 1 — Conv × 2 → Pooling | 필터 64개
model.add(Conv2D(64,kernel_size=(2,2),padding='same',activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(64,kernel_size=(2,2),padding='same',activation='relu'))  # Conv 2/13
model.add(Dropout(0.3))  
model.add(MaxPool2D(pool_size=(2,2)))  # Pooling: 32×32 → 16×16, 채널 64개 유지  # Conv 1/13


# Block 2 — Conv × 2 → Pooling | 필터 128개
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))  # Conv 3/13
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))  # Conv 4/13
model.add(MaxPool2D(pool_size=(2,2)))  # Pooling: 16×16 → 8×8, 채널 128개 유지
model.add(Dropout(0.2))  


# Block 3 — Conv × 3 → Pooling | 필터 256개
model.add(Conv2D(256,(2,2),padding='same',activation='relu'))  # Conv 5/13
model.add(Conv2D(256,(2,2),padding='same',activation='relu'))  # Conv 6/13
model.add(Conv2D(256,(2,2),padding='same',activation='relu'))  # Conv 7/13
model.add(MaxPool2D(pool_size=(2,2)))  # Pooling: 8×8 → 4×4, 채널 256개 유지
model.add(Dropout(0.3))  


# 분류기 — Flatten → Dense → Dropout → Dense → Dropout → Dense(100)
model.add(GlobalAveragePooling2D())  # 1×1×512 특징맵 → 512개 값의 벡터
model.add(Dense(512, activation='relu'))  # Dense 1/3: 추출한 특징 조합
model.add(Dropout(0.3))  
model.add(Dense(512, activation='relu'))  # Dense 2/3: 분류에 필요한 특징 학습
model.add(Dropout(0.2))  
model.add(Dense(100, activation='softmax'))  # Dense 3/3: 100개 클래스의 확률 출력
model.summary()
# exit()

#3 컴파일,훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics= ['acc']
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train,y_train,
          epochs=1000,
          verbose=1,
          batch_size=128,
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
#1
loss:  3.166536808013916
acc:  0.21089999377727509
accuracy_score :  0.2109
걸린시간 : 583.76 초

#2
loss:  2.940565586090088
acc:  0.24529999494552612
accuracy_score :  0.2453

#3
accuracy_score :  0.3334
걸린시간 : 899.52 초

#4
accuracy_score :  0.3439
걸린시간 : 1079.55 초

loss:  2.625156879425049
acc:  0.3409000039100647
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.3409


9 - acc: 0.3545
loss:  2.579921007156372
acc:  0.3544999957084656
313/313 [==============================] - 6s 18ms/step
accuracy_score :  0.3545


loss:  2.562361478805542
acc:  0.3752000033855438
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.3752
걸린시간 : 376.18 초

loss:  2.486510992050171
acc:  0.3840000033378601
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.384
걸린시간 : 408.77 초

loss:  2.3752896785736084
acc:  0.4120999872684479
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.4121
걸린시간 : 812.73 초

model.add(GlobalAveragePooling2D())  추가후 변화
loss:  2.0546157360076904
acc:  0.46889999508857727
accuracy_score :  0.4689
걸린시간 : 763.13 초

cnn-> dnn
oss:  1.5248686075210571
acc:  0.46209999918937683
313/313 [==============================] - 1s 3ms/step
accuracy_score :  0.4621

loss:  1.4667251110076904
acc:  0.4832000136375427
313/313 [==============================] - 1s 3ms/step
accuracy_score :  0.4832
걸린시간 : 1125.71 초
'''
