import os
import numpy as np
import pandas as pd
import argparse
from urllib.request import urlretrieve
import struct as st
import gzip
import math

class LoadMNIST:

    def __init__(self):
        self.url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        self.url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        self.url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        self.url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

        self.trainX = self.load_images(self.url_train_image)
        self.trainY = self.load_labels(self.url_train_labels)
        self.testX = self.load_images(self.url_test_image)
        self.testY = self.load_labels(self.url_test_labels)

    def load_labels(self,url):
        train_image_file = self.openfile(url)
        train_image_file.read(8)
        return np.frombuffer(train_image_file.read(),dtype = np.uint8)

    def load_images(self,url):
        train_image_file = self.openfile(url)
        train_image_file.read(16)
        return np.frombuffer(train_image_file.read(), dtype=np.uint8).reshape(-1, 28*28)

    def openfile(self,url):
        local_filename, headers = urlretrieve(url)
        return gzip.open(local_filename, 'rb')

class NaiveBayes:
    def __init__(self, data):
        self.data = data
        print("Load")
        self.discrete_MLE =  self.discrete(self.data.trainX, self.data.trainY)
        print("MLE")
        self.prior = self.cal_prior(self.data.trainY)
        print('prior')
        print(self.prior)
        self.discrete_posterior, self.discrete_error = self.pred(self.prior,self.discrete_MLE,self.data.testX,self.data.testY,True)
        print(self.discrete_error)
        '''
        self.cont_MLE =  self.continuous(self.data.trainX, self.data.trainY,self.data.testX)
        self.cont_posterior, self.cont_error = self.pred(self.prior,self.cont_MLE,self.data.testX,self.data.testY,False)
        '''
    def discrete(self, images, labels):
        count_labels = [0.0 for i in range(len(images))]
        count_imagesbins = [[[1 for i in range(32)] for j in range(len(images[0]))] for k in range(10)]
    
        for i in range(len(images)):
            count_labels[labels[i]] += 1
            for j in range(len(images[0])):
                # numerator is 256 not 255 for [0,1)
                count_imagesbins[labels[i]][j][int(images[i][j]/256 *32)] +=1

        prob_images_bins = [[[count_imagesbins[i][j][k]/count_labels[i] for k in range(32)] for j in range(len(images[0]))] for i in range(10)]
        return prob_images_bins

    def cal_prior(self,labels):
        count_labels = [0.0 for i in range(len(labels))]
        for i in range(len(labels)):
            count_labels[labels[i]] += 1
        prob_labels = [count_labels[i] / len(labels) for i in range(10)]
        return prob_labels

    def normal_pdf(self,x,mean, var):
        if var ==0:
            return 1
        return exp(-(x-mean)^2/(2*var))/(2*np.pi*var)


    def continuous(self, images, labels,testX):
        mean = [[0.0 for i in range(10)] for j in range(len(images[0]))]
        var = [[0.0 for i in range(10)] for j in range(len(images[0]))]
        MLE = [[0.0 for i in range(10)] for j in range(len(images[0]))]
        for i in range(10):
            for j in range(len(images[0])):
                tmp = [x for x in images[:][j] if labels[i]==i]
                mean[i][j] = np.array(tmp).mean()
                var[i][j] = np.array(tmp).var()
                MLE[i][j] = self.normal_pdf(testX[:][j],mean[i][j],var[i][j])
        return MLE


    def pred(self,prior,MLE, testX, testY,discrete):
        predict_class = [0.0 for i in range(len(testX))]
        posterior = [[0.0 for i in range(10)] for j in range(len(testX))]
        for i in range(len(testX)):
            sum = -1
            max = -math.inf
            for j in range(10):
                if discrete:
                    sum = np.array([MLE[j][k][int(testX[i][k] / 256 * 32)] for k in range(len(testX[0]))])
                    sum=np.log(sum)
                    sum=np.sum(sum,axis=0)

                else:
                    sum+= np.array([MLE[j][k] for k in range(len(testX[0]))]).sum()
                '''
                for k in range(len(testX[0])):
                    if discrete:
                        if MLE[j][k][int(testX[i][k] / 256 * 32)] != 0:
                            sum += MLE[j][k][int(testX[i][k] / 256 * 32)]
                    else:
                        if MLE[j][k] != 0:
                            sum += MLE[j][k]
                '''
                sum += prior[j]
                posterior[i][j] = sum
                if sum > max:
                    max = sum
                    predict_class[i] = j
        predict_error = len([i for i in range(len(testX)) if predict_class[i] != testY[i]]) / len(testX)
        return posterior, predict_error

class Beta_binomial:
    def __init__(self,txt,a,b):
        self.init_a =a
        self.init_b =b
        self.trial = opener(txt)
        online_learning(a,b,self.trial)

    def opener(self,txt):
        file = open(txt,'r')
        return file.readlines()

    def online_learning(self,init_a,init_b,trial):
        parameter_a = [init_a for i in range(len(trial)+1)]
        parameter_b = [init_b for i in range(len(trial)+1)]
        print('initial parameter a={} , b ={}'.format(init_a,init_b))
        for i in trial:
            n = len(i)
            m=0
            count_1 = i.split('0')
            for j in count_1:
                m += len(j)
            parameter_a[i+1] = parameter_a[i] + m
            parameter_b[i+1] = parameter_b[i] + n -m
            print('Trial_{} :'.format(i+1))
            print('MLE = {}'.format(m/n))
            print('prior, a={} , b ={}'.format(parameter_a[i], parameter_b[i]))
            print('posterior, a={} , b ={}'.format(parameter_a[i+1], parameter_b[i+1]))



def main():

    parser = argparse.ArgumentParser(description = 'Naive method')
    parser.add_argument('--toggle',default = 0, type = int, help =' discrete 0, continuous 1')
    parser.add_argument('--a',default = 3, type = int, help =' parameter a for the initial beta prior')
    parser.add_argument('--b',default = 5, type = int, help =' parameter b for the initial beta prior')
    args = parser.parse_args()
    
    data = LoadMNIST()
    naivebayes = NaiveBayes(data)
    '''

    if args.toggle ==0:
        print()
    online_learning = Beta_binomial('test.txt',args.a,args.b)
    '''



if __name__ == "__main__":
    main()



