import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams['axes.unicode_minus'] = False 
# 如果目录下没有iris_training.csv这个文件就要用注释语句中的部分下载数据
# TRAIN_URL="http://download.tensorflow.org/data/iris_training.csv"
# train_path=tf.keras.utils.get_file(TRAIN_URL.split('/')[0],TRAIN_URL)
# df_iris=pd.read_csv(train_path,header=0)
df_iris=pd.read_csv(r"C:\Users\ASUS1\.keras\datasets\iris_trainning.csv")
dftest_iris=pd.read_csv(r"C:\Users\ASUS1\.keras\datasets\iris_test.csv")
testiris=np.array(dftest_iris)
iris=np.array(df_iris)
#取出花瓣宽度，花瓣长度，花萼长度，花萼宽度
train_x=iris[:,[3,2,0,1]]
train_y=iris[:,4]
test_x=testiris[:,[3,2,0,1]]
test_y=testiris[:,4]
#取出变色鸢尾和维基利亚鸢尾
x_train=train_x[train_y>0]
y_trian=train_y[train_y>0]
x_test=test_x[test_y>0]
y_test=test_y[test_y>0]
# # 可视化
# cm_pt=mpl.colors.ListedColormap(["blue","red"])
# plt.scatter(x_train[:,0],x_train[:,1],c=y_trian,cmap=cm_pt)
# plt.show()
# x_train[:,0]=(x_train[:,0]-tf.reduce_min(x_train[:,0]))/(tf.reduce_max(x_train[:,0])-tf.reduce_min(x_train[:,0]))
# x_train[:,1]=(x_train[:,1]-tf.reduce_min(x_train[:,1]))/(tf.reduce_max(x_train[:,1])-tf.reduce_min(x_train[:,1]))

num=len(x_train)
for i in range(0,num):
    if y_trian[i]==2.0:
        y_trian[i]=0.0
nn=len(x_test)
# print(nn)
for i in range(0,nn):
    if y_test[i]==2.0:
        y_test[i]=0.0
x_train=x_train-np.mean(x_train,axis=0)
x_test=x_test-np.mean(x_test,axis=0)

# print(y_test)
x0_train=np.ones(num).reshape(-1,1)
X=tf.cast(tf.concat((x0_train,x_train),axis=1),tf.float32)
Y=tf.cast(y_trian.reshape(-1,1),tf.float32)
x0_train=np.ones(nn).reshape(-1,1)
xt=tf.cast(tf.concat((x0_train,x_test),axis=1),tf.float32)
yt=tf.cast(y_test.reshape(-1,1),tf.float32)

# print(yt)
#设置超参数
learn_rate=0.1
iter=300
display_step=30
np.random.seed(612)
W=tf.Variable(np.random.randn(5,1),dtype=tf.float32)
ce=[]
acc=[]
testce=[]
testacc=[]
for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        PRED=1/(1+tf.exp(-tf.matmul(X,W)))
        Loss=-tf.reduce_mean(Y*tf.math.log(PRED)+(1-Y)*tf.math.log(1-PRED))
        PREDtest=1/(1+tf.exp(-tf.matmul(xt,W)))
        Losstest=-tf.reduce_mean(yt*tf.math.log(PREDtest)+(1-yt)*tf.math.log(1-PREDtest))
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED.numpy()<0.5,0.0,1.0),Y),tf.float32))
    accuracytest=tf.reduce_mean(tf.cast(tf.equal(tf.where(PREDtest.numpy()<0.5,0.0,1.0),yt),tf.float32))

    ce.append(Loss)
    acc.append(accuracy)
    testce.append(Losstest)
    testacc.append(accuracytest)

    dl_dw=tape.gradient(Loss,W)
    W.assign_sub(learn_rate*dl_dw)
    if i%display_step==0:
        print("i:%i,trainAss:%f,trainLoss:%f,testAss:%f,testLoss:%f"%(i,accuracy,Loss,accuracytest,Losstest))


plt.figure(figsize=(6,3))
plt.subplot(121)
plt.title("训练集损失+准确率")
plt.plot(ce,color="b",label="Loss")
plt.plot(acc,color="r",label="acc")
plt.legend()


plt.subplot(122)
plt.title("测试集损失+准确率")
plt.plot(testce,color="b",label="testLoss")
plt.plot(testacc,color="r",label="testacc")
plt.legend()
plt.tight_layout()
plt.show()