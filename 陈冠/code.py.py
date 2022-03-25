# 哈哈哈，傻逼
import numpy as np
import pandas as pd
class Lei:
    def __init__(self,x,y):#初始化
        self.x=x
        self.y=y
    def qiub(self): #求系数的函数
        xt=self.x.T #矩阵x的转置矩阵xt
        xtx=np.dot(xt,x)  #xt乘以x
        xtx_inv=np.linalg.inv(xtx)#求xtx的逆矩阵
        xtx_invxt=np.dot(xtx_inv,xt)
        self.b=np.dot(xtx_invxt,self.y)#求b
        return self.b
    def returnb(self):
        return self.b
    def predict(self,x):  #预测函数
        y=np.dot(x,self.b)
        return y
x=pd.read_csv('X.csv')
y=pd.read_csv('Y.csv')
#X=df[]
#Y=df['outcome'
#print(X.shape)
#print(Y.shape)
#print(Y[Y.isnull()])
#print(Y.isnull())
#df=pd.read_csv('train_.csv')
#df=df.dropna
#print(df)
#df['outcome'].dropna(axis=0,inplace=True)
#print(df[df['outcome'].isnull()])
'''df=df.drop([12,36,38,43,48,64,74,346])
print(df[df['outcome'].isnull()])'''
#print(df.loc([len(1)],['id']))
'''df2=df.dropna()
df2.to_csv('train2.csv')'''
'''ones=np.ones(len(X))
X=np.c_[ones,X]'''
lei=Lei(x,y)
b=lei.qiub()
print(b)
x=pd.read_csv('test_.csv')
sample=lei.predict(x)#获得预测值
print(sample)
print(len(sample))
ab=pd.read_csv('sample_submission.csv')
print(len(ab))#样本数量相等



