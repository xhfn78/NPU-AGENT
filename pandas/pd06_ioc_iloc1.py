import numpy as np
import pandas as pd
print(pd.__version__) #3.0.6

data = [
    ['삼성','1000','2000'],
    ['현대','1100','3000'],
    ['LG','2000','500'],
    ['아모레','3500','6000'],
    ['네이버','100','1500'],
]

index = ['031','059','033','045','023']
columns = ['종목명','시가','종가']

df = pd.DataFrame(data=data,index=index,columns=columns)
# print(df)

#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

#인덱스와 컬럼명은 연산에 관계 없음!!!!!!!!!

# print(df[0]) #keyError(key) 라고 뜸 사용불가
# print(df['031']) #key 에러뜸
# print(df['종목명']) # !!!!!'판다스 열행'!!!!! 컬럼이 기준

########아모레 시가 출력###############
# print(df[3,1]) #에러
# print(df[3][1]) #에러

# print(df['045','종목명']) #KeyError: ('045', '종목명')


# print(df['045']['종목명']) #KeyError

print(df['종목명']['045']) #아모레 !@!!행,열 에서 열,행으로 바꾸면 출력가능
# !!!!!'판다스 열행'!!!!!
########################################
#loc : 인덱스 기준으로 행 데이터 추출
#iloc : 행 번호를 기준으로 행 데이터 추출
        #!!!!!! int loc로 외워라 !!!!!!!!!!!!

########################################

# print('=============아모레 뽑기=============')
# print(df.iloc[3]) #아모레의 행이 나옴
# 종목명     아모레  
# 시가     3500
# 종가     6000
# Name: 045, dtype: str

# print(df.iloc['045']) #Cannot index by location index with a non-integer key
# print(df.loc[3]) #KeyError: 3
# print(df.loc['045'])
# 종목명     아모레
# 시가     3500
# 종가     6000
# Name: 045, dtype: str

# print('=============네이버 뽑기=============')
# print(df.iloc[4]) # iloc 가 붙으면 인덱스 숫자로 뽑아내야함
# print(df.loc['023'])# loc가 붙으면 인덱스의 값을 넣어야함 '023'문자열로 

print('=============아모레 종가 뽑기=============')
# print(df.iloc[3][2]) key 에러 #pandas 2버전대에서는 가능
# print(df.iloc[3,2]) #6000 
# print(df.iloc[3]['종가']) #6000 이렇게 쓰는건 가능
# print(df.iloc[3,'종가']) #이렇게는 불가능!

# print(df.loc['045'][2]) #KeyError: 2
# print(df.loc['045',2]) ##KeyError: 2
print(df.loc['045']['종가']) #6000
print(df.loc['045','종가']) #6000

#기본 판다스는 열행기준!
#행열 계산을 하고싶으면 loc,iloc 를 쓰면  가능@!!@!!

print('=========================')

print(df.iloc[3].iloc[2]) #6000
print(df.iloc[3].loc['종가']) #6000

print(df.loc['045'].loc['종가'])#6000
print(df.loc['045'].iloc[2])  #6000