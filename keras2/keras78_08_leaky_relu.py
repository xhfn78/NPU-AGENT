import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
alpha = 0.01


# def leaky_relu(x):
    # return np.maximum(alpha*x, x)
    # return np.where(x>0,x,alpha*x)

leaky_relu= lambda x : np.where(x>0,x,alpha*x)
y =leaky_relu(x) 

plt.plot(x,y)
plt.grid()
plt.show()


#relu에서 0이하의 값을 무시해보니까 뭔가 찜찜해서 음수도 살려봄