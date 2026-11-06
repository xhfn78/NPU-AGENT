# [실습] 과적합 확인 - 당뇨병 데이터셋
# hist에 담긴 loss와 val_loss를 같이 그려서, val_loss가 언제부터 올라가는지 본다.
from sklearn.datasets import fetch_california_housing, load_diabetes #캘리포니아 집값 데이터셋,로드 디아벳
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Dropout
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
import time
#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=42
)

#2. 모델구성
def build_model(drop=0.5,optimizer ='adam',activation='relu',
                node1=128,node2=64,node3=32,node4=16,node5=8,
                lr= 0.001 
                ):
    inputs = Input(shape=(10,), name='inputs1')
    x = Dense(node1,activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2,activation=activation,name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3,activation=activation,name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4,activation=activation,name='hidden4')(x)
    x = Dense(node5,activation=activation,name='hidden5')(x)
    outputs = Dense(1, activation='linear',name='outputs')(x)

    model = Model(inputs=inputs,outputs=outputs)

    model.compile(optimizer=optimizer,metrics=['mae'],loss='mse')
    return model 

def create_hyperparameter():
    batchs = [32,16,8,1,64]
    optimizers = ['adam','rmsprop','adadelta']
    dropouts =[0.2,0.3,0.4,0.5]
    activations = ['relu','elu','selu','linear']
    node1 = [128,64,32,16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
    node4 = [128,64,32,16]
    node5 = [128,64,32,16]
    return {
        'batch_size': batchs,
        'optimizer': optimizers,
        'drop' : dropouts,
        'activation': activations,
        'node1': node1,
        'node2': node2,
        'node3': node3,
        'node4': node4,
        'node5': node5,

    }

hyperparameters = create_hyperparameter()
print(hyperparameters)


from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model,verbose=1,)
model = RandomizedSearchCV(keras_model,
                           hyperparameters,
                           cv=3,
                           random_state=333,
                           n_iter=5,
                           n_jobs=2,
                           verbose=1,
                           )


from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=8,
    verbose=1,
    factor=0.8

)




#3.훈련
start = time.perf_counter()
model.fit(x_train,y_train,epochs=1000,verbose=1,callbacks=[es,rlr,],
          validation_split=0.1,)
end = time.perf_counter()

print('최적의 매개변수:', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

#4.평가,예측

print('model.score:', model.score(x_test,y_test))
print('best_score:', model.best_score_) #train data에 대한 score  과적합 스코어임
print('걸린시간:',round(end-start,2),'초')



# print('결과값: ' ,results)
#랜덤 442
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2857.6667 
# loss: 2857.666748046875
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 결과값:  [[210.36603 ]

#랜덤 221
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2459.2292 
# loss: 2459.229248046875
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 

#랜덤 3333
# 42/42 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2882.3848 
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 3071.2979 
# loss: 3071.2978515625
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 결과값:  [[204.20064 ]
