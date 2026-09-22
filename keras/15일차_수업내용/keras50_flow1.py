from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

path = 'c:/study/_data/image/'

img = load_img(path + '내사진.png', target_size =(150,150))

print(img)

#<PIL.Image.Image image mode=RGB size=150x150 at 0x1AE292B70D0> 랩핑된 데이터 

# plt.imshow(img)
# plt.show()


arr = img_to_array(img)
# print(arr.shape) #(150, 150, 3)
# print(type(arr)) #<class 'numpy.ndarray'>

arr = np.expand_dims(arr,axis=0) #arr를 0열에 넣는다 =차원증가 3차원--> 4차원
# print(arr.shape)  #(1, 150, 150, 3) 4차원 shape로 변형

# np_path = './_data/kaggle_cat_dog_npy/'

# np.save(np_path + 'keras49_me.npy', arr= arr)

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

it = datagen.flow(arr,  #arr을 datagen속 파라미터 들을 배치 사이즈 만큼 랜덤 적용 현재 1번 
             batch_size=1,
             )
# print(it)  # 이터레이터 = 리스트랑 비슷하다

# print(it.next()) #파이썬 3.10까지, 이렇게씀
# print(nex t(it).shape) #파이썬 3.11이후 


fig,ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))

for i in range(5):
    batch = next(it)
    print(batch.shape)
    batch = batch.reshape(150,150,3)

    ax[i].imshow(batch)
plt.show()    