import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,5)



# def softmax(x):
#     return np.exp(x)/ np.sum(np.exp(x))
    
softmax= lambda x : np.exp(x)/ np.sum(np.exp(x))
y =softmax(x)

ratio = y 
labels = y 


plt.pie(ratio,labels,shadow=True,startangle=90)
plt.grid()
plt.show()
#relu에서 0이하의 값을 무시해보니까 뭔가 찜찜해서 음수도 살려봄