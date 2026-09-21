#44-1 카피
import numpy as np
from tensorflow.python.keras.layers import GlobalAveragePooling2D,MaxPool2D,Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


path = './_save/cat_dog/'  


# model = load_model(path + 'keras29_1_save_model.keras') #저장된 모델 불러오기
model = load_model(path +'catdog_save_1.keras' ) 


#평가예측
np_path = './_data/kaggle_cat_dog_npy/'

me= np.load(np_path + 'keras49_me.npy',)
me = me/255.


print('==============================')
y_pred = model.predict(me)
print(y_pred)  
y_pred = np.round(y_pred)


print(y_pred)

