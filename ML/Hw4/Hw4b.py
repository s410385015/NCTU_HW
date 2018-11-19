import numpy as np
import os
from PIL import Image
from math import *
from struct import *

"""
Read MNIST Dataset from file path
flag = true  : image data -> return 2d array m[index][pixel]
                (eg. 60000 train data with 28*28 image will return m[60000][28*28])
flag = false : label data -> return 1d array l[index]
                (eg. 60000 label data will return l[60000])
"""
def ReadMNIST(path,flag=True):
    path=os.path.split(os.path.realpath(__file__))[0]+'\\'+path
    file=open(path,'rb')
    data=[]
    if flag:
        #First 16 bytes of image data are header, so drop it
        file.read(4)
        img_num=unpack('>I',file.read(4))
        file.read(8)
        bits=28*28*img_num[0]
        bitsString = '>' + str(bits) + 'B'
        data=unpack_from(bitsString,file.read(), 0)
        file.close()
        data=np.reshape(data,[img_num[0],28*28])
        return data.astype(int)
    else :
        #First 8 bytes of label data are header, so drop it
        file.read(4)
        bits=unpack('>I',file.read(4))
        bitsString = '>' + str(bits[0]) + 'B'
        data=unpack_from(bitsString,file.read(), 0)
        
        return np.array(data).astype(int)

def main():

    '''
    train [img_num,pixels]
    label [img_num,label]
    '''
    image_train=ReadMNIST('train-images.idx3-ubyte')
    label_train=ReadMNIST('train-labels.idx1-ubyte',False)
    print(np.shape(image_train))
    print(np.shape(label_train))


   

    #Binarize the img
    image_train[image_train<=127]=0
    image_train[image_train>127]=1
  

    img_num,dim=np.shape(image_train)
    class_num=10

    count=np.zeros(class_num)
    for i in range(label_train.size):
        count[label_train[i]]+=1

    _lambda=np.random.dirichlet(np.ones(10))
    #_lambda=np.ones(10)/10
    pre_lambda=_lambda
    p=np.random.rand(class_num,dim)
    p=p/2+0.25
    w=np.empty([img_num,class_num])
    
    
    _p=np.reshape(p*255,(28*class_num,28))
    total=30
    for epoch in range(total):
        #E step
        #calculate w
    
        tmp=np.dot(image_train,np.log(p.T))+np.dot(1-image_train,np.log(1-p.T))
        tmp[:]+=_lambda
        w=np.exp(tmp)  
        w/=(np.sum(w,axis=1).reshape(len(w),1))
       
        
        #M step
       
    
        _lambda=np.sum(w,axis=0)/img_num
        
        r=np.dot(w.T,image_train)
        q=np.sum(w,axis=0).reshape(class_num,1)
        p=r/q
        p[p<1e-50]=1e-50
        _lambda[_lambda<1e-5]=1e-5


        #epoch log
        print("----------------")
        print("iteration: "+str(epoch))
        print(_lambda)
        error=np.sum(np.abs(pre_lambda-_lambda))
        pre_lambda=_lambda
        print("error: "+str(error))

        if error <0.0005:
            break


        tmp_p=np.reshape(p*255,(28*class_num,28))
        _p=np.concatenate((_p,tmp_p),axis=1)
        
        
        
    
    

    img = Image.fromarray(_p)
    img=img.convert('RGB')
    img.save('result.png')
    print(count/img_num)

    classify=np.argmax(w,axis=1)
    table=np.zeros((class_num,class_num))
    _table=np.zeros((class_num,class_num))

    '''
    for i in range(img_num):
        if classify[i]==label_train[i]:
            table[label_train[i]][1]+=1
        else:
            table[classify[i]][0]+=1
    '''
    table=table.astype(int)
    _table=table.astype(int)

    for i in range(img_num):
        _table[classify[i]][label_train[i]]+=1
    

    print(_table)
    row=input()
    row=row.split(" ")

    for i in range(len(row)):
        table[int(row[i])]=_table[i]
        
    print(table)
    print(count)

    for i in range(class_num):
        print("-----------")
        print(str(i))
        TP=table[i][i]
        FN=count[i]-TP
        FP=np.sum(table[i])-TP
        TN=img_num-count[i]-FP
        sensitivity=TP/(TP+FN)
        specificity=FP/(FP+TN)
        print(np.array([[TP,FP],[FN,TN]]))
        print("sensitivity: "+str(sensitivity))
        print("specificity: "+str(specificity))
        


if __name__ =="__main__":
    main()