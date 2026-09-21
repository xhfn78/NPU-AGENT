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

#2. 모델구성

model = Sequential()
model.add(Conv2D(64, kernel_size=(10,10), input_shape=(100,100,3,),activation='relu',padding='same' ))
# model.add(Conv2D(64, kernel_size=(5,5), activation='relu',padding='same' ))
model.add(Dropout(0.2))
model.add(MaxPool2D(2,2))

model.add(Conv2D(128,(4,4),activation='relu' , padding='same'))
# model.add(Conv2D(128,(4,4),activation='relu' , padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))


model.add(Conv2D(256,(2,2),activation='relu' ,padding='same' ))
# model.add(Conv2D(32,(2,2),activation='relu' ,padding='same' ))
# model.add(MaxPool2D(2,2))
# model.add(Dropout(0.2))




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