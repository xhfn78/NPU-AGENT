#44-1 카피
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D,MaxPool2D,Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1.데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,  #수평 뒤집기,
    # vertical_flip=True,    #수직 뒤집기(상하 반전)
    # width_shift_range=0.1, #평형이동 
    # height_shift_range=0.1, #
    # rotation_range=5,  #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,
    # shear_range=0.7, #좌표하나를 고정하고 다른 몇개의 좌표로 이동(한마디로 찌부)
    # fill_mode='neareet',
)


test_datagen = ImageDataGenerator(
    rescale=1./255,  #테스트 데이터는 원형으로 유지
)
path_train = './_data/image/cat_dog/training_set/'  # train속 ad와 normal은 라벨링해줌
path_test = './_data/image/cat_dog/test_set/'

xy_train = train_datagen.flow_from_directory(  
    path_train, #경로
    target_size=(200,200),  #이미지를 크기를 (100,100)으로 만들어줌 원하는 크기 가능!
    batch_size=10000,
    class_mode='binary',  #이진분류
    color_mode='rgb', #흑백
    shuffle=True,
)
#Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200,200),  #이미지를 크기를 (100,100)으로 만들어줌 원하는 크기 가능!
    batch_size=10000,
    class_mode='binary',  #이진분류
    color_mode='rgb', #흑백
    shuffle=False,  # 테스트에서는 필요없음
)
 #Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
 
print(x_train.shape,y_train.shape) #(5000, 200, 200, 3) (5000,)
print(x_test.shape,y_test.shape) #(2023, 200, 200, 3) (2023,)

# exit()
#2. 모델구성

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(200,200,3,),activation='relu',))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64, (2,2), activation='relu',))
model.add(MaxPool2D(2,2))

model.add(Conv2D(128, (2,2), activation='relu',))
model.add(MaxPool2D(2,2))

model.add(Conv2D(256, (2,2), activation='relu',))
model.add(MaxPool2D(2,2))

model.add(Conv2D(512, (2,2), activation='relu',))
model.add(MaxPool2D(2,2))
model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(units=130,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16,activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.summary()

# exit()
#3 컴파일,훈련
model.compile(loss='binary_crossentropy', 
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
          epochs=100,
          batch_size=16,
          verbose=1,
          validation_split =0.2,
          callbacks =[es,],
          )
end_time = time.time()

#평가예측
print(('=====================model.evaluate=================='))
loss = model.evaluate(x_test,y_test)
print('==============================')
print("loss:", loss[0])
print('acc:',round(loss[1],4)) #loss: 0번[0.12450382113456726,###LOSS값 (1번) 0.9473684430122375]###ACC값
print('==============================')
y_pred = model.predict(x_test)  #시그모이드 함수를 거쳐 0,1사이 값을 반환후 >>metrics=['acc']로 후처리하면 0 OR 1로 반올림내림해서 퍼센테이지로 변환
y_pred = np.round(y_pred)# y_pred한 값이 0.11121515,0.125148이런식으로 나와서 라운드처리후  [1.] 이런식으로 변환한다음에 acc값 비교 이거 안하면 에러남
# 0.5를 기준으로 반올림한다. 0.5 이상이면 1, 미만이면 0.
# accuracy_score는 정수 라벨끼리 비교하는 함수라서 확률값을 그대로 넣으면 에러가 난다.
acc_score = accuracy_score(y_test,y_pred)
print('accuracy_score : ', acc_score)
print('걸린시간 :',round(end_time-start_time,2),'초')



# ==============================
# loss: 0.5341812372207642
# acc: 0.7494
# ==============================

#  acc: 0.7884
# ==============================
# loss: 0.4743961691856384
# acc: 0.7884
# ==============================

# ==============================
# loss: 0.4654271900653839
# acc: 0.7924
# ==============================

# ==============================
# loss: 0.4213293194770813
# acc: 0.825
# ==============================