#50-2 카피

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten,GlobalAveragePooling2D,MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


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
randidx = np.random.randint(x_train.shape[0], size=augment_size) #60000개중 40000개 랜덤뽑기
# print(randidx)  #[15000  3128 45286 ... 54111 32130 21793]
# print(randidx.shape) #(40000,) 
# print(len(randidx)) #40000
# print(np.min(randidx),np.max(randidx))

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# print(x_augmented.shape,y_augmented.shape) 
#(40000, 28, 28) (40000,) 아직 데이터변형은 되지않음

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],1,)  #
# print(x_augmented.shape) #(40000, 28, 28, 1)


xy_augmented = datagen.flow(
    x_augmented,y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]
 #### 변환 완료 #########
# print(x_augmented.shape) # (40000, 28, 28, 1)
# print(x_train.shape)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

x_train = np.concatenate((x_train,x_augmented))
y_train = np.concatenate((y_train,y_augmented))

# print(x_train.shape) #(100000, 28, 28, 1)
# print(y_train.shape) #(100000,)

# print(np.unique(y_train, return_counts=True,))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
#  array([ 9979, 10011, 10127,  9976,  9935,  9835, 10082, 10055, 10014,




# print(x_train.shape)    #(60000, 28, 28)
# print(x_train[0].shape) #(28, 28)

aaa = np.tile(x_train[0],augment_size).reshape(-1,28,28,1) #(100, 28, 28, 1)
# print(aaa.shape) #(28, 2800)

xy_data = datagen.flow(
        np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),
        np.zeros(augment_size),
        batch_size=augment_size,
        shuffle=False,

).next()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)  
y_test = y_test.reshape(-1,1) 
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape,y_test.shape)  #(60000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32,(5,5),input_shape = (28,28,1),padding='same'))  #  (26 ,26,64)
# model.add(Conv2D(filters=64, kernel_size=(5,5),activation='relu')) #(24,24,32)
model.add(Dropout(0.2))
model.add(MaxPool2D(2,2))

# model.add(Conv2D(128,(2,2),activation='relu')) #(None, 23, 23, 32)
model.add(Conv2D(64,(4,4),activation='relu')) #(None, 22, 22, 16)
model.add(Dropout(0.2))
model.add(MaxPool2D(2,2))

model.add(Conv2D(128,(2,2),activation='relu')) #(None, 22, 22, 16)
model.add(Conv2D(128,(2,2),activation='relu')) #(None, 22, 22, 16)
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(units=16, activation='relu')) #아웃풋 node의 갯수= units
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))#(None, 10) 
model.summary()

#3 컴파일,훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics= ['acc']
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train,y_train,
          epochs=1000,
          batch_size=256,
          verbose=2,
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

313/313 [==============================] - 1s 2ms/step - loss: 0.4061 - acc: 0.9055
loss:  0.40610525012016296
acc:  0.9054999947547913
313/313 [==============================] - 1s 1ms/step
accuracy_score :  0.9055
걸린시간 : 583.39 초

313/313 [==============================] - 1s 2ms/step - loss: 0.3130 - acc: 0.9114
loss:  0.3129764795303345
acc:  0.9114000201225281
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9114
걸린시간 : 528.02 초

loss:  0.30824926495552063
acc:  0.9117000102996826
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9117
걸린시간 : 364.06 초


'''