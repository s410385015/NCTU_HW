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

   



'''
create a lookup table of probability of the value in each pixel for each label class
data[num_image][num_pixel] -> prob[num_pixel][num_label_class][value_range]
eg.
    num_image = 2           -> prob[4][2][4]
    num_pixel = 2*2                 prob[0][0]=[0,1,0,0] prob[0][1]=[1,0,0,0]
    value_range 0~3                 prob[1][0]=[0,1,0,0] prob[1][1]=[1,0,0,0]
    data[0]=[1,1,3,2]=class 0       prob[2][0]=[0,0,0,1] prob[2][1]=[0,0,1,0]
    data[1]=[0,0,3,2]=class 1       prob[3][0]=[0,0,1,0] prob[3][1]=[0,1,0,0]
    label class=2               
'''
def createProbTable(data,label,class_number,value_range):
    
    #peudocount 
    prob=np.zeros((len(data[0]),class_number,value_range))+1
    data=(data/8).astype(int)
 

    for i in range(class_number):
        dis=np.where(label==i)[0]
        dis=data[dis]
        for j in range(len(data[0])):
            index,count=np.unique(dis[:,j],return_counts=True)
            prob[j][i][index]=count[:]

    return prob



'''
implement the Naive Bayes Classify
The category of each image is the one having the highest posterior
posterior can be calculate by
p(n)=p( pixel1 | class n)*p(pixel2 | class )*.....p(pixel 28*28 | class n)*p(class n)
'''
def Classifier(count,prob,test,total):
    predict=np.zeros(len(test)).astype(int)
    test=(test/8).astype(int)

    score=np.zeros((len(test),len(count)))
    #for k in range(len(count)):
        #prob[:][k][:]=(prob[:][k][:]/count[k])
        
    for i in range (len(test)):
        max=-inf
        pdt_class=-1
        for k in range(len(count)):
        
            
            tmp=prob[np.arange(len(test[i])),k,test[i]]
            tmp=tmp/count[k]
            tmp=np.log(tmp)
            sum=np.sum(tmp,axis=0)
            #sum=np.prod(tmp)
            sum+=log(count[k]/total)
            score[i][k]=sum
            if sum>max:
                max=sum
                pdt_class=k
        predict[i]=pdt_class
    
    return predict,score


'''
Calculate the mean & variance of each pixel in each class
'''
def CalculateMeanAndVariance(data,label,count):
    sum=np.zeros((len(count),len(data[0])))
    mean=np.zeros((len(count),len(data[0])))
    variance=np.zeros((len(count),len(data[0])))
   

    for i in range(len(count)):
        dst=np.where(label==i)[0]
        dst=data[dst]
        #sum[i]=np.sum(dst,axis=0)
        #mean[i]=sum[i]/count[i]
        mean[i]=np.mean(dst,0)
        variance[i]=np.var(dst,0)
    
  

    

    return mean,variance   


'''
Calculate the posterior with the Gaussion
'''
def ClassifierWithGaussion(mean,variance,test,total,count):
      
    predict=np.zeros(len(test)).astype(int)
    score=np.zeros((len(test),len(count)))

    for i in range (len(test)):
        max=-inf
        pdt_class=-1
        for k in range(len(count)):

            index=[variance[k]!=0]
            g=Gaussion(mean[k][index],variance[k][index],test[i][index])
            #g=Gaussion(mean[k][index],variance[k][index],test[i][index])
            _g=g[g>0]
            _g=np.log(_g)
            sum=np.sum(_g,axis=0)
            sum+=log((count[k]/total))
            score[i][k]=sum
            if sum>max:
                max=sum
                pdt_class=k
        predict[i]=pdt_class
    
    return predict,score


'''
input : mean , variance and x
output : the gaussion value based on formula -> (1/(2*pi*variance^2)^(1/2))*e^(-((x-mean)^2/2*variance^2)) 
'''
def Gaussion(m,v,x):
    
    return ((1/np.sqrt(2*pi*v))*(e**(-((x-m)**2)/(2*v))))


'''
print the score of each row
eg. # 0 1 2 3 4 5 6 7 8 9
    1 a b c d e f g h i j
    2 a b c d e f g h i j
'''

def WriteScore(path,score):
    path=os.path.split(os.path.realpath(__file__))[0]+'\\'+path
    file=open(path,"w")
    file.write("#,0,1,2,3,4,5,6,7,8,9\n")
    file.close()
    file=open(path,"a")
    for i in range(len(score)):
        s=str(i)
        for j in range(len(score[i])):
            s+=","+str(score[i][j])
        file.write(s+'\n')
    file.close()

def main():

  
    class_num=10

    image_train=ReadMNIST('train-images.idx3-ubyte')
    label_train=ReadMNIST('train-labels.idx1-ubyte',False)
    image_test=ReadMNIST('t10k-images.idx3-ubyte')
    label_test=ReadMNIST('t10k-labels.idx1-ubyte',False)
    
    mode=input("Type in mode (0 for discrete mode, 1 for continuous mode.) : ")
    
    #calculate the prior
    count=np.zeros(class_num)
    
    for i in range(label_train.size):
        count[label_train[i]]+=1
    print("Training label:")
    print(count)
    total=len(label_train)
    prior=count/total
    print("Prior:")
    print(prior)

    result=False
    # discrete mode,
    if mode == '0':
        
        print("Building Prob table ...")
        # thr value of pixel will map from 0~255 into 0~31
        prob_table=createProbTable(image_train,label_train,class_num,32)
        print("Done!")

        print("Starting predict ...")
        predict,score=Classifier(count,prob_table,image_test,total)
        print("Done!")
        result=True
        WriteScore("Discrete_Score.csv",score)
    # continuous mode
    elif mode == '1':

        print("Calculating mean & variance ...")
        # thr value of pixel will map from 0~255 into 0~31
        mean,variance=CalculateMeanAndVariance(image_train,label_train,count)
        print("Done!")

        print("Starting predict ...")
        predict,score=ClassifierWithGaussion(mean,variance,image_test,total,count)
        print("Done!")
        result=True
        WriteScore("Continuous_Score.csv",score)
    if result == True:
        
        print("Write score to file ...")
        
        count=[0 for i in range(class_num)]
        total=len(predict)
        for i in range(total):
            count[predict[i]]+=1

        print("Predict:")
        print(count)
        posterior=[round(count[i]/total,3) for i in range(class_num)]
        print("Posterior:")
        print(posterior)


        match=0
        for i in range(len(label_test)):
            if label_test[i]==predict[i]:
                match+=1
        print("Correct Predict:")
        print(match)

        print("Error rate:")
        error=round(1-((match)/len(label_test)),3)
        print(error)


if __name__ =="__main__":
    main()