x = 10
y = 10
w =0.001
lr =0.999
epochs = 1000

for i ,e in enumerate(range(epochs)):
    hypothesis = x * w
    loss = (hypothesis -y) **2 #mse, 바이어스,시그마  신경안쓴 상태

    print(i+1 ,'loss :', round(loss,4)+100, '\t Predict:', round(hypothesis,4))
    up_predict = x*(w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict = x*(w - lr)
    down_loss = (y - down_predict) ** 2

    if (up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr

