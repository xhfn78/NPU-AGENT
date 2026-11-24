#sliu는 다른말로 swish라고함

#x에 sigmoid() 곱한 함수
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)


def silu(x):
    return x * (1/(1+np.exp(-x)))


silu(x) = lambda x : x * (1/(1+np.exp(-x)))

y =silu(x)

plt.plot(x,y)
plt.grid()
plt.show()


#relu에서 0이하의 값을 무시해보니까 뭔가 찜찜해서 음수도 살려봄