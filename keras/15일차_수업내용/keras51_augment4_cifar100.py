import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# x_train = x_train /225.0
# x_test = x_test /225.0
#############데이터 증폭#########################
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,  #수평 뒤집기,
    # vertical_flip=True,    #수직 뒤집기(상하 반전)
    # width_shift_range=0.1, #평형이동 
    height_shift_range=0.1, #
    # rotation_range=5,  #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,
    # shear_range=0.7, #좌표하나를 고정하고 다른 몇개의 좌표로 이동(한마디로 찌부)
    fill_mode='nearest',
)

augment_size = 40000  
# print(x_train.shape[0])
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False) #60000개중 40000개 랜덤뽑기
# print(randidx)  #[15000  3128 45286 ... 54111 32130 21793]
# print(randidx.shape) #(40000,) 
# print(len(randidx)) #40000
# print(np.min(randidx),np.max(randidx))

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape,y_augmented.shape) 
#(40000, 28, 28) (40000,) 아직 데이터변형은 되지않음
# exit()
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3,)  #
print(x_augmented.shape) #(40000, 28, 28, 1)


xy_augmented = datagen.flow(
    x_augmented,y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]
 #### 변환 완료 #########
print(x_augmented.shape) # (40000, 28, 28, 1)
print(x_train.shape)
# exit()
x_train = x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)

x_train = np.concatenate((x_train,x_augmented))
y_train = np.concatenate((y_train,y_augmented))

#######################원한 인코더###########################
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)  
y_test = y_test.reshape(-1,1) 
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
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
# model.add(Conv2D(256,(2,2),padding='same',activation='relu'))  # Conv 7/13
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
'''
