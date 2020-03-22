import numpy as np 

np.random.seed(612)
rando=np.random.rand(1,1000)
inputnumber=int (input("plese input a number:"))
n=1
print("序号","索引值","随机数")
for i in range(1000):
    if i%inputnumber==0:
        print(n,i,rando[0][i])
        n=n+1
