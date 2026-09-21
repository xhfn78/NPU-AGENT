from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

path = 'c:/study/_data/image/'

img = load_img(path + '개.png', target_size =(100,100))

print(img)

#<PIL.Image.Image image mode=RGB size=150x150 at 0x1AE292B70D0> 랩핑된 데이터 

# plt.imshow(img)
# plt.show()


arr = img_to_array(img)
# print(arr.shape) #(150, 150, 3)
# print(type(arr)) #<class 'numpy.ndarray'>




arr = np.expand_dims(arr,axis=0) #arr를 0열에 넣는다 =차원증가 3차원--> 4차원
# print(arr.shape)  #(1, 150, 150, 3) 4차원 shape로 변형

np_path = './_data/kaggle_cat_dog_npy/'

np.save(np_path + 'keras49_me.npy', arr= arr)