'캘리포니아','디아벳'
import numpy as np
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing,load_diabetes
from sklearn.metrics import r2_score

from sklearn.utils import all_estimators
import sklearn as sk
import warnings
import time
warnings.filterwarnings('ignore')  #warning 있는 거 무시하고 진행

#1. data

cali= fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [cali,diabetes]

data_name = ['캘리','디아벳']



#2.모델
# all = all_estimators(type_filter='classifier',)
all = all_estimators(type_filter='regressor',)
n_split = 5

kFold = KFold(n_splits=n_split,shuffle=True,random_state=333)

start = time.time()
for index,value in enumerate(datasets):
    x,y =value
    x_train,x_test, y_train,y_test = train_test_split(
    x,y ,shuffle=True,random_state=333,train_size=0.8,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print('=========',data_name[index],'================')
    model_name=[]
    model_r2=[]

    for name,model in all:
        # print(name)
        try:    
            model = model()

            #.3훈련
            # model.fit(x_train,y_train)
            scores = cross_val_score(
                model,x_train,y_train, cv=kFold
            )
            #4.평가
            y_pred = cross_val_predict(model,x_test,y_test,cv=kFold)

            r2 = np.round(r2_score(y_test,y_pred),4)
            model_name.append(name)
            model_r2.append(r2)
        
          
            
        except:
            # print(name,'모델 오류')
            continue
    
    model_r2 = np.array(model_r2)
    max_index =np.argmax(model_r2)
    print('최고모델:',model_name[max_index],'r2',model_r2[max_index])    
end = time.time()


print('걸린시간:',round(end-start,2),'초')