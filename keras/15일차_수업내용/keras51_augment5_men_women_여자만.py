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

np_path = './_data/faces_npy/'
x_train = np.load(np_path + 'faces_01_x_train.npy', ) 
y_train =np.load(np_path + 'faces_01_y_train.npy', ) 
x_test = np.load(np_path + 'faces_01_x_test.npy', ) 
y_test = np.load(np_path + 'faces_01_y_test.npy',) 



x_train_women  = x_train[np.where(y_train > 0.0)]
y_train_women  = y_train[np.where(y_train > 0.0)]


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
unique_counts = np.unique(y_train,return_counts=True,)[1]
augment_size = unique_counts[0]- unique_counts[1]

# print(augment_size)
# exit()
# print(x_train.shape[0])
randidx = np.random.choice(x_train.shape[0], size=augment_size,replace=False) #60000개중 40000개 랜덤뽑기
print(randidx)  #[15000  3128 45286 ... 54111 32130 21793]
# print(randidx.shape) #(40000,) 
print(len(randidx)) #40000
print(np.min(randidx),np.max(randidx))

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# print(x_augmented.shape,y_augmented.shape) 
# (40000, 28, 28) (40000,) 아직 데이터변형은 되지않음

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3,)  #
# print(x_augmented.shape) #(40000, 28, 28, 1)

xy_augmented = datagen.flow(
    x_augmented,y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]


print(x_augmented.shape) # (40000, 28, 28, 1)
print(x_train.shape)
# exit()
x_train = x_train.reshape(-1,100,100,3)
x_test = x_test.reshape(-1,100,100,3)

x_train = np.concatenate((x_train,x_augmented))
y_train = np.concatenate((y_train,y_augmented))

# exit()
#2. 모델구성

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(100,100,3,),activation='relu',padding='same' ))
# model.add(Conv2D(64, kernel_size=(5,5), activation='relu',padding='same' ))
model.add(Dropout(0.2))
model.add(MaxPool2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu' , padding='same'))
# model.add(Conv2D(128,(4,4),activation='relu' , padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(256,(2,2),activation='relu' ,padding='same' ))
model.add(Conv2D(256,(2,2),activation='relu' ,padding='same' ))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())
# model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.summary()

#3 컴파일,훈련
model.compile(loss='binary_crossentropy', 
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
          epochs=2000,
          batch_size=32,
          verbose=1,
          validation_split =0.2,
          callbacks =[es,],
          )
end_time = time.time()
path = './_save/man_women/'  
model.save(path + 'man_women_save_13.keras') #모델 세이브


#4.평가예측
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
# loss: 0.24654521048069
# acc: 0.9017
# =========================