import numpy as np
import os
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

    _lambda=np.random.dirichlet(np.ones(10))
    p=np.random.rand(class_num,dim)
    p=p/2+0.25
    w=np.empty([img_num,class_num])
    
    print("lambda")
    print(_lambda)

    for epoch in range(10):
        #E step
        #calculate w
        for c in range(class_num):
            for n in range(img_num):
                tmp=(p[c,:]**image_train[n,:])*((1-p[c,:])**(1-image_train[n,:]))
                tmp=np.log(tmp)
                tmp=np.sum(tmp)
                w[n][c]=exp(tmp+log(_lambda[c]))
        for n in range(img_num):
            w[n,:]/=(np.sum(w[n,:]))
           


        
        #M step
        for c in range(class_num):
            _lambda[c]=np.sum(w[:,c])/img_num
        for d in range(dim):
            for c in range(class_num):
                p[c][d]=max(np.sum(w[:,c]*image_train[:,d])/np.sum(w[:,c]),1e-50)
        #p[p<1e-50]=1e-50
        _lambda[_lambda<1e-5]=1e-5
        print(_lambda)
        
  
if __name__ =="__main__":
    main()