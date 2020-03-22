import numpy as np 

x=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
x_=x.mean()
y_=y.mean()
x_x_=x-x_
x_2=x_x_**2
y_y_=y-y_
fenm=x_x_*y_y_
fsum=np.sum(fenm)
msum=np.sum(x_2)
w=fsum/msum
b=y_-w*x_
print("w的值是：{}      b的值是：{}".format(w,b))