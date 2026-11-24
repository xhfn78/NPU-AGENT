import numpy as np


aaa= np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T # #컬럼이 2개임


def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data,[25,50,75]) #aaa의 데이터의 25프로 50프로 70프로 지점

    # print('1사분위:',quartile_1)
    # print('q2:',q2)
    # print('3사분위:',quartile_3)
    iqr = quartile_3 - quartile_1

    print('IQR:',iqr)

    lower_bound = quartile_1 - (iqr *1.5) #최솟ㅏㅓ값
    upper_bound = quartile_3 + (iqr *1.5) #최댓값
    return np.where((data > upper_bound) | (data < lower_bound)), \
                iqr,lower_bound,upper_bound

for i in range(aaa.shape[1]):
    indices, iqr, lower, upper = outlier(aaa[:, i])

    aaa[:, i] = np.clip(aaa[:, i], lower, upper)

print(aaa)

    # print('기준 범위:', lower, '~', upper)
    # print('이상치 행 인덱스:', indices[0])
    # print('이상치 값:', column[indices])

# import matplotlib.pyplot as plt

# plt.boxplot(aaa)
# plt.axhline(up, color='pink',label = 'Upper Bound')
# plt.axhline(low, color='pink',label = 'lower Bound')
# plt.legend()
# plt.show()