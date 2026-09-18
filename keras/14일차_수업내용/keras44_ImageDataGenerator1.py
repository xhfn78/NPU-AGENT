import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,  #수평 뒤집기,
    vertical_flip=True,    #수직 뒤집기(상하 반전)
    width_shift_range=0.1, #평형이동 
    height_shift_range=0.1, #
    rotation_range=5,  #각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=1.2,
    shear_range=0.7, #좌표하나를 고정하고 다른 몇개의 좌표로 이동(한마디로 찌부)
    fill_mode='neareet',
)


test_datagen = ImageDataGenerator(
    rescale=1./255,  #테스트 데이터는 원형으로 유지
)
path_train = './_data/image/brain/train'  # train속 ad와 normal은 라벨링해줌
path_test = './_data/image/brain/test'

xy_train = train_datagen.flow_from_directory(  
    path_train, #경로
    target_size=(100,100),  #이미지를 크기를 (100,100)으로 만들어줌 원하는 크기 가능!
    batch_size=10,
    class_mode='binary',  #이진분류
    color_mode='grayscale', #흑백
    shuffle=True,
)
#Found 160 images belonging to 2 classes.
xy_train = test_datagen.flow_from_directory(
    path_test,
    target_size=(150,150),  #이미지를 크기를 (100,100)으로 만들어줌 원하는 크기 가능!
    batch_size=10,
    class_mode='binary',  #이진분류
    color_mode='grayscale', #흑백
    shuffle=False,  # 테스트에서는 필요없음
)
 #Found 120 images belonging to 2 classes.

print(xy_train) #<keras.preprocessing.image.DirectoryIterator object at 0x0000016DFEC29060>
# print(xy_train.next())  #이터레이터의 첫번째 출력
# print(xy_train.next())  #이터레이터의 두번째 출력  배치사이즈가 총 １０이니까 총사진 갯수８０ 개라서 ８개가 들어있음

# print(xy_train[0])   # 첫번쨰 배치의 x데이터
# print(xy_train[0][1])  # 첫번쨰 배치의 y데이터
print(xy_train[0][0].shape) #(10, 150, 150, 1)
print(xy_train[0][1].shape)# (10,)

# print(xy_train[15][0])  #16하면 에러남 노멀,ad 사진 갯수가 각80,80개라 합치면 160장


print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class  o'numpy.ndarray'>






