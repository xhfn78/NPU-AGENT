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
path_train = './_data/image/brain/train/'  # train속 ad와 normal은 라벨링해줌
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(  
    path_train, #경로
    target_size=(150,150),  #이미지를 크기를 (100,100)으로 만들어줌 원하는 크기 가능!
    batch_size=160,
    class_mode='binary',  #이진분류
    color_mode='grayscale', #흑백
    shuffle=True,
)
#Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150,150),  #이미지를 크기를 (100,100)으로 만들어줌 원하는 크기 가능!
    batch_size=120,
    class_mode='binary',  #이진분류
    color_mode='grayscale', #흑백
    shuffle=False,  # 테스트에서는 필요없음
)
 #Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
 
# print(x_train.shape,y_train.shape) #(160, 150, 150, 1) (160,)
# print(x_test.shape,y_test.shape) #(120, 150, 150, 1) (120,)


np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path + 'keras45_01_x_train.npy', ) #x
y_train =np.load(np_path + 'keras45_01_y_train.npy', ) #x
x_test = np.load(np_path + 'keras45_01_x_test.npy', ) #x
y_test = np.load(np_path + 'keras45_01_y_test.npy',) #x
