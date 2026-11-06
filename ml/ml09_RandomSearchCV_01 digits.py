import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBRFClassifier
import time
import warnings
warnings.filterwarnings('ignore')

#1데이터 

x, y = load_digits(return_X_y=True)


x_train,x_test,y_train,y_test = train_test_split(
    x,y ,shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True ,random_state=333)
learning_rate = [0.1,0.05,0.01,0.005,0.001]
max_depth = [3,4,5,6,7,8,]



parameters = [
    {'n_estimators': [100,500],'max_depth':[6,10,12],
    'learning_rate':[0.1,0.01,0.001]},  #18번
    
    {'max_depth':[6,8,10,12] ,'learning_rate':[0.1,0.01,0.001]}, #12번
    {'min_child_weight':[2,3,5,10] ,'learning_rate':[0.1,0.01,0.001]}, #12번
    #42번 돈다 
]


#2.모델


xgb = XGBRFClassifier()
model = RandomizedSearchCV(xgb,parameters,cv=kfold, # x5 해서 총 210번 돌아감
             n_jobs=-1,
             verbose=1,
             random_state=100, #무작위로 뽑는 파라미터 조합을 재현하기 위한 설정
             refit=True,#교차검증에서 가장 좋은 성능을 보인 파라미터 조합을 찾은 다음, 
             #.fit()에 전달한 학습 데이터 전체로 모델을 새로 학습합니다.
             )

#3.훈련
start = time.perf_counter()
model.fit(x_train,y_train)
end = time.perf_counter()

print('최적의 매개변수:', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

#4.평가,예측

print('model.score:', model.score(x_test,y_test))
print('best_score:', model.best_score_) #train data에 대한 score  과적합 스코어임
print('걸린시간:',round(end-start,2),'초')

# 최적의 매개변수: XGBRFClassifier(base_score=None, booster=None, callbacks=None,
#                 colsample_bylevel=None, colsample_bytree=None, device=None,
#                 early_stopping_rounds=None, enable_categorical=False,
#                 eval_metric=None, feature_types=None, feature_weights=None,
#                 gamma=None, grow_policy=None, importance_type=None,
#                 interaction_constraints=None, learning_rate=0.1, max_bin=None,
#                 max_cat_threshold=None, max_cat_to_onehot=None,
#                 max_delta_step=None, max_depth=12, max_leaves=None,
#                 min_child_weight=None, missing=nan, monotone_constraints=None,
#                 multi_strategy=None, n_estimators=100, n_jobs=None,
#                 num_parallel_tree=None, objective='multi:softprob', ...)
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 100}
# model.score: 0.9305555555555556
# best_score: 0.9415408439798683
# 걸린시간: 26.77 초


# 최적의 파라미터 : {'max_depth': 12, 'learning_rate': 0.1}
# model.score: 0.9305555555555556
# best_score: 0.9415408439798683
# 걸린시간: 12.83 초

# 최적의 파라미터 : {'max_depth': 12, 'learning_rate': 0.1}
# model.score: 0.9305555555555556
# best_score: 0.9415408439798683
# 걸린시간: 33.42 초
# PS C:\study> 

# 최적의 파라미터 : {'max_depth': 12, 'learning_rate': 0.1}
# model.score: 0.9305555555555556
# best_score: 0.9415408439798683
# 걸린시간: 12.77 초