import numpy as np
from sklearn.datasets import load_iris,load_wine,load_breast_cancer,load_digits
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk
import warnings
import time
warnings.filterwarnings('ignore')  #warning 있는 거 무시하고 진행

#1. data

iris = load_iris(return_X_y=True)
wine = load_wine(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris,cancer,wine,digits]

data_name = ['아이리스','캔서','와인','디지트']



#2.모델
all = all_estimators(type_filter='classifier',)
# all = all_estimators(type_filter='regressor',)
n_split = 5

kFold = StratifiedKFold(n_splits=n_split,shuffle=True,random_state=333)

start = time.time()
for index,value in enumerate(datasets):
    x,y =value
    x_train,x_test, y_train,y_test = train_test_split(
    x,y ,shuffle=True,random_state=333,train_size=0.8,stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print('=========',data_name[index],'================')
    model_name=[]
    model_acc=[]

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

            acc = np.round(accuracy_score(y_test,y_pred),4)
            model_name.append(name)
            model_acc.append(acc)
        
          
            
        except:
            # print(name,'모델 오류')
            continue
    
    model_acc = np.array(model_acc)
    max_index =np.argmax(model_acc)
    print('최고모델:',model_name[max_index],'acc',model_acc[max_index])    
end = time.time()


print('걸린시간:',round(end-start,2),'초')


# ========= 아이리스 ================
# 최고모델: BaggingClassifier acc 0.9667
# ========= 캔서 ================
# 최고모델: AdaBoostClassifier acc 0.9825
# ========= 와인 ================
# 최고모델: MLPClassifier acc 1.0
# ========= 디지트 ================
# 최고모델: ExtraTreesClassifier acc 0.9556
# 걸린시간: 124.06 초