import pandas as pd
import numpy as np
import os
import re
import time
#创建一个空DataFrame columns = [0,1...1023,''type]
# a = []
# for i in range(1024):
#     a.append(i)
# a.append('type')
# df = pd.DataFrame(columns=a)
#
# # 训练样本中有0-9 之间的数字
# for j in range(10):
#     lists ={}
#     for i in range(250):
#         path = r'./digits/trainingDigits/{}_{}.txt'.format(j,i)
#           #拼接路径，因为每个数字的样本数量不一样，所以使用try 捕获异常
#         try:
#             data = np.loadtxt(path,dtype='str')
#             lists['0_'+str(i)]=[int(i) for i in  ''.join(data)]
#             # 使每一个样本的内容成一行，每一个数字独占一个属性
#         except:
#             pass
#         df0 = pd.DataFrame(lists)
#         df0=df0.T
#         df0['type']=str(j)
#     df=pd.concat([df,df0],ignore_index=True,axis=0)
#         # 把所有样本拼接成一个大的模型并根据样本名给每一行添加上属于数字几
#
# df.to_excel('trainmodol.xlsx')
# 保存到本地，下次打开可以节约时间

# 训练模型
# trainmodol=pd.read_excel('trainmodol.xlsx')
# print(trainmodol)





class traing():
    def __init__(self):
        self.trainmodol = pd.read_excel('trainmodol.xlsx')
        self.text=self.trainmodol.iloc[:,:-1]  #读取模型中除了type列的所有属性
    def Tr(self,data,num,k):
        self.trainmodol['d'] = np.sum((self.text-data)**2,axis=1)**(1/2)   #计算相似度并添加到表中
        self.trainmodol.sort_values('d',inplace=True)   #根据相似度升序排列
        type = self.trainmodol[['type']].astype('category')   #转成category格式方便统计频率最高的type
        # print('原数字:',num,'识别后:',self.frequency(type))
        if num==int(self.frequency(type,k)):
            #判断识别结果与传入进去的样本是否一样
            return 1
        else:
            return 0

    def frequency(self, df,k):
        df = (df.head(k)).describe()
        #取k值，排序后的前K个，统计那个type出现的频率高，TOP就是识别后的结果
        return df.iloc[2,0]



txtlist = os.listdir(r'./digits/testDigits/')
name = {}

for i in range(10):
    pt=r'{}_(\d*)\.txt'.format(i)
    type = re.findall(pt,''.join(txtlist))
    name[i]=len(type)

a = traing()
for k in range(50,230):
    starttime = time.time()
ta =0
for j in name:
    for i in range(name[j]):
        path = r'./digits/testDigits/{}_{}.txt'.format(j,i)
        try:
            data = np.loadtxt(path,dtype='str')
            numb=j
            data =[int(i) for i in  ''.join(data)]
            data=np.array(data)
            td = a.Tr(data,j,51)
            ta+=td   #识别结果正确，ta就加一
        except:
            pass
    stoptime = time.time()


    # kev = str(k)+","+str(ta/945)
    # with open('k值.txt','a') as f:
    #     f.write(kev+'\n')
    print('当k为:{},准确度:{}'.format(k,ta/945))
    print('用时:',stoptime-starttime)

