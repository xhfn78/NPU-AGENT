#44-1 카피
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D,MaxPool2D,Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
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

path_train = './_data/image/rps/'  # train속 ad와 normal은 라벨링해줌


xy_train = train_datagen.flow_from_directory(  
    path_train, #경로
    target_size=(150,150),  #이미지를 크기를 (100,100)으로 만들어줌 원하는 크기 가능!
    batch_size=160,
    class_mode='categorical',  
    color_mode='rgb', 
    shuffle=True,
)

x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, random_state=333, stratify=xy_train[0][1],)
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


np_path = './_data/rps_npy/'
np.save(np_path + 'keras45_01_x_train.npy', arr = x_train) 
np.save(np_path + 'keras45_01_y_train.npy', arr = y_train) 
np.save(np_path + 'keras45_01_x_test.npy', arr = x_test) 
np.save(np_path + 'keras45_01_y_test.npy', arr = y_test) 
exit()
