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

np_path = './_data/rps_npy/'
x_train = np.load(np_path + 'keras45_01_x_train.npy', ) 
y_train =np.load(np_path + 'keras45_01_y_train.npy', ) 
x_test = np.load(np_path + 'keras45_01_x_test.npy', ) 
y_test = np.load(np_path + 'keras45_01_y_test.npy',) 

#2. 모델구성

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=(150,150,3,),activation='relu', ))
model.add(Dropout(0.2))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu' , ))
model.add(Dropout(0.2))
model.add(MaxPool2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu' , ))
model.add(MaxPool2D(2,2))

model.add(Conv2D(256,(2,2),activation='relu' , ))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())
# model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(16,activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))
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

#평가예측
print(('=====================model.evaluate=================='))
loss = model.evaluate(x_test,y_test)
print('==============================')
print("loss:", loss[0])
print('acc:',round(loss[1],4)) #loss: 0번[0.12450382113456726,###LOSS값 (1번) 0.9473684430122375]###ACC값
print('==============================')
y_predict = np.argmax(model.predict(x_test), axis=1)
y_actual = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_actual,y_predict,normalize=True)
print('acc_score:', acc_score)  #acc_score: 0.9298245614035088
acc_score = accuracy_score(y_actual,y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 :',round(end_time-start_time,2),'초')