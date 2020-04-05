import tensorflow as tf
import numpy as np

load_data=np.loadtxt("商品房销售记录表.csv",delimiter=",",skiprows=2)
sess=tf.Session()

x1=tf.constant(load_data[:,1],tf.float32)
x2=tf.constant(load_data[:,2],tf.float32)
y=tf.constant(load_data[:,3],tf.float32)


x0=tf.ones(sess.run(tf.size(x1)),tf.float32)
x=tf.stack([x0,x1,x2],axis=0)
x=tf.transpose(x)
y=tf.reshape(y,[-1,1])
xt=tf.transpose(x)
xtx_1=tf.linalg.inv(tf.matmul(xt,x))
xtx_1_xt=tf.matmul(xtx_1,xt)

w=tf.matmul(xtx_1_xt,y) 

while 1:
    area=float(input("请输入面积：20-500之间的实数："))
    number=int(input("请输入房间数：1-10之间的整数："))
    if area>=20 and area<=500 and number>=1 and number<=10:
        print("预估房价："+str(sess.run(sess.run(w[1])*area+w[2]*number+sess.run(w[0]))))
    else:
        print("输入错误")