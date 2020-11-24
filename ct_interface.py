import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import ttk
import os
from os import walk
from torchvision import transforms 
import torch.nn as nn
import torch.nn.functional as f
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import glob
import cv2
import torchvision.transforms.functional as TF
from PIL import Image, ImageTk
import os

from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
import tensorflow as tf
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.layers.core import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from PIL import Image
import numpy as np
import random
import cv2
import os
import numpy as np
import cv2 as cv2
import math
np.seterr(divide='ignore', invalid='ignore')
#loaading data
from skimage.feature import greycomatrix, greycoprops
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import time
import csv

torch.cuda.get_device_name(0)
torch.cuda.is_available()



def HTF_Energy(m): ###################    f1
    f = []
    lists = []
    for i in m:
        for j in i:
            j=j**2
            #print("%.14f"%j)
            f.append(j)
    global showFeatureValue
    #if (showFeatureValue):
        #print("Energy : "+str(sum(f)))
    return sum(f)

def HTF_Entropy_Correlation_Inverse_Inertia_SumAvg_ClusterShade_ClusterProminence(mat): ################## f2
    f = []
    lenM = int(math.sqrt(len(mat[0])))

    
    global glcmlex
    #print(mat.shape)
    Zeros = 0.00000000000000001
    
    mu=0        
    var=0

    ClusterShade,ClusterProminence,Correlation = 0,0,0   
    Entropy = 0
    Inverse = 0
    Inertia = 0
    SumAvg2 = 0
    log = 0
    for a in range(lenM):
        for b in range(lenM):                
            mu+= a*mat[a][b]
    #print("mu : "+str(mu))  
    for a in range(lenM):
        for b in range(lenM):
            var += ((a-mu)**2) * mat[a][b]
            
    for i in range(lenM):
        for j in range(lenM):
            Correlation += ((i-mu + Zeros) * (j-mu+Zeros) * ((mat[i][j])+Zeros))
            Inverse += ((mat[i][j]+Zeros) / (1+(i-j)**2) + Zeros)
            Inertia += ((i-j)**2) * (mat[i][j]+Zeros)
            SumAvg = (i*(mat[i][j]+Zeros)) + (j*(mat[i][j])+Zeros)                
            SumAvg2 +=i*(SumAvg+Zeros)
            ClusterShade  += (((i-mu) + (j-mu))**3) * (mat[i][j]+Zeros)
            ClusterProminence  += (((i-mu) + (j-mu))**4) * (mat[i][j]+Zeros)
            
            if mat[i][j] == 0:
                log = 0+Zeros
                
                #print("%.9f"%s)                    
            else:
                log = math.log2(mat[i][j])
            Entropy += ((mat[i][j]+Zeros)* log)                                                        
    
        
        
    global showFeatureValue
    #if (showFeatureValue):
        #print("Entropy : "+str(-result))
    Correlation = Correlation /( (var**2) + Zeros)
    return -Entropy,Correlation,Inverse,Inertia,SumAvg2,ClusterShade,ClusterProminence









def HTF_Haralick(mat):###################    f9
    f = 0
    lenM = int(math.sqrt(len(mat[0])))
    global glcmlex
    Zeros = 0.00000000000000001
    #mat = np.reshape(mat,(glcmlex,glcmlex))
    mu = 0           
    muForvar=[]
    result = 0
    muT=0
    for a in range(lenM):
        mu=a*mat[a]            
        mu=sum(mu)
        muT+=mu                     
    for i in range(lenM):     
        for j in range(lenM):                
            result += ((((i*j) *(mat[i][j])+Zeros ) -  muT**2 +Zeros))
            #s=((((i*j) *(mat[i][j] + Zeros) ) - muT**2))
            #d=i*j
            #g=d*mat[i][j]                
            #print("%.8f"%s)               
    var = 0
    for a in range(lenM):
        for b in range(lenM):
            var += ((a-muT)**2) * mat[a][b]                
    
    #print(result)
    f = result / ((var**2)+Zeros)
    global showFeatureValue
    #if (showFeatureValue):

        #print("Haralick : "+str(f))
    return f
def dowload_test_data(namePath):
    datas = []
    i= 0
    
    namePath += '/'
    
    for (d,c,path) in os.walk (namePath):
            
        for image in path:
            i+=1
            
            
            data = namePath+image
            datas.append(data)
                #print(image)
            #for (d,c,path) in os.walk(image):
    #print(i)
    #print(len(datas))
    #print(datas[1])
    return datas


from matplotlib import pyplot as plt


import csv

def CSV_GLCM_RGB(DATA):
    #print(name,len(DATA))
    #print(DATA)
    iter = 0
    result = []
    for i in DATA:
        
        iter += 1
        #if iter %100== 0:
            
            #print("Iteration : "+str(iter)+"  TIME : "+str(time.time()-start))
            
        dataset = []        
        #print(i)
        
        #print(img)
        img2 = cv2.imread(i)
        r = img2[:,:,0]
        g = img2[:,:,1]
        b = img2[:,:,2]
       # print(img.shape)
        
        r = r//32
        g = g//32
        b = b//32

        split = i.split('/')
        #print(split[-1])
        dataset.append(split[-1])

       
        for i in range(0,4):
            glcm1 = greycomatrix(r, distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/45], levels=8,
                        symmetric=True, normed=True)
            norm1 = glcm1[:,:,0,i:i+1].reshape(8,8)
            f1 = HTF_Energy(norm1)
            f2,f3,f4,f5,f6,f7,f8 = HTF_Entropy_Correlation_Inverse_Inertia_SumAvg_ClusterShade_ClusterProminence(norm1)
            f9 = HTF_Haralick(norm1)
            dataset.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9])
        
            
            glcm2 = greycomatrix(g, distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/45], levels=8,
                        symmetric=True, normed=True)
            norm2 = glcm2[:,:,0,i:i+1].reshape(8,8)
            f1 = HTF_Energy(norm2)
            f2,f3,f4,f5,f6,f7,f8 = HTF_Entropy_Correlation_Inverse_Inertia_SumAvg_ClusterShade_ClusterProminence(norm2)
            f9 = HTF_Haralick(norm2)
            dataset.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9])


            
            glcm3 = greycomatrix(b, distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/45], levels=8,
                        symmetric=True, normed=True)            
            norm3= glcm3[:,:,0,i:i+1].reshape(8,8)
            f1 = HTF_Energy(norm3)
            f2,f3,f4,f5,f6,f7,f8 = HTF_Entropy_Correlation_Inverse_Inertia_SumAvg_ClusterShade_ClusterProminence(norm3)
            f9 = HTF_Haralick(norm3)
            dataset.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9])

            
            
            
        
        
        result.append(dataset)
        
    return result
def create_Model_RGB(weights_path):
    
    model = Sequential()
    #model.add(Conv1D(filters=36, kernel_size = 2, activation ='relu',input_shape=(36,1) ))
    model.add(Conv1D(filters=64, kernel_size = 2, activation ='relu' ,input_shape=(108,1)))
    #model.add(Conv1D(filters=64, kernel_size = 2, activation ='relu' ))
    model.add(Conv1D(filters=64, kernel_size = 2, activation ='relu' ))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=128, kernel_size = 2, activation ='relu' ))
   
    model.add(Conv1D(filters=128, kernel_size = 2, activation ='relu' ))
    model.add(MaxPooling1D(pool_size=2))

   

    model.add(Conv1D(filters=256, kernel_size = 2, activation ='relu' ))
    
    model.add(Conv1D(filters=256, kernel_size = 2, activation ='relu' ))
    
    model.add(Conv1D(filters=256, kernel_size = 2, activation ='relu' ))
    model.add(MaxPooling1D(pool_size=2))

    """
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))
    
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))

    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))
    model.add(MaxPooling1D(pool_size=2))

    
    
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))
 
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))
   
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))   
    model.add(MaxPooling1D(pool_size=2))"""
              
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5)) 
    
    model.add(Dense(3,activation='sigmoid') )

    if weights_path:
        model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print("model created")
    return model

def CSV_GLCM_grey(DATA):
    #print(name,len(DATA))
    #print(DATA)
    iter = 0
    result = []
    for i in DATA:
        
        iter += 1
        #if iter %100== 0:
            
        #    print("Iteration : "+str(iter)+"  TIME : "+str(time.time()-start))
            
        dataset = []        
        #print(i)
        img = cv2.imread(i,0)
        #print(img)
        #img2 = cv2.imread(i)
        #r = img2[:,:,0]
        #g = img2[:,:,1]
        #b = img2[:,:,2]
        #print(img.shape)
        img = img // 32
        """r = r//8
        g = g//8
        b = b//8"""

        split = i.split('/')
        #print(split[-1])
        dataset.append(split[-1])

       
        for i in range(0,4):
            """r = greycomatrix(r, distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/45], levels=8,
                        symmetric=True, normed=True)
            norm1 = glcm[:,:,0,i:i+1].reshape(8,8)
            f1 = HTF_Energy(norm)
            f2,f3,f4,f5,f6,f7,f8 = HTF_Entropy_Correlation_Inverse_Inertia_SumAvg_ClusterShade_ClusterProminence(norm)
            f9 = HTF_Haralick(norm)
            dataset.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9])
        
            
            g = greycomatrix(g, distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/45], levels=8,
                        =True, normed=True)
            norm2 = glcm[:,:,0,i:i+1].reshape(8,8)
            f1 = HTF_Energy(norm)
            f2,f3,f4,f5,f6,f7,f8 = HTF_Entropy_Correlation_Inverse_Inertia_SumAvg_ClusterShade_ClusterProminence(norm)
            f9 = HTF_Haralick(norm)
            dataset.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9])


            
            b = greycomatrix(b, distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/45], levels=8,
                        symmetric=True, normed=True)            
            norm3= glcm[:,:,0,i:i+1].reshape(8,8)
            f1 = HTF_Energy(norm)
            f2,f3,f4,f5,f6,f7,f8 = HTF_Entropy_Correlation_Inverse_Inertia_SumAvg_ClusterShade_ClusterProminence(norm)
            f9 = HTF_Haralick(norm)
            dataset.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9])"""

            
            glcm = greycomatrix(img, distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/45], levels=8,
                        symmetric=True, normed=True)
            norm = glcm[:,:,0,i:i+1].reshape(8,8)
        
        
            f1 = HTF_Energy(norm)
            f2,f3,f4,f5,f6,f7,f8 = HTF_Entropy_Correlation_Inverse_Inertia_SumAvg_ClusterShade_ClusterProminence(norm)
            f9 = HTF_Haralick(norm)
            dataset.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9])
            
        
        
        result.append(dataset)
        
    return result
def create_Model_Greyscale(weights_path):
    
    model = Sequential()
    #model.add(Conv1D(filters=36, kernel_size = 2, activation ='relu',input_shape=(36,1) ))
    model.add(Conv1D(filters=64, kernel_size = 2, activation ='relu' ,input_shape=(36,1)))
    #model.add(Conv1D(filters=64, kernel_size = 2, activation ='relu' ))
    model.add(Conv1D(filters=64, kernel_size = 2, activation ='relu' ))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=128, kernel_size = 2, activation ='relu' ))
   
    model.add(Conv1D(filters=128, kernel_size = 2, activation ='relu' ))
    model.add(MaxPooling1D(pool_size=2))

   

    model.add(Conv1D(filters=256, kernel_size = 2, activation ='relu' ))
    
    model.add(Conv1D(filters=256, kernel_size = 2, activation ='relu' ))
    
    model.add(Conv1D(filters=256, kernel_size = 2, activation ='relu' ))
    model.add(MaxPooling1D(pool_size=2))

    """
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))
    
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))

    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))
    model.add(MaxPooling1D(pool_size=2))

    
    
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))
 
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))
   
    model.add(Conv1D(filters=512, kernel_size = 2, activation ='relu' ))   
    model.add(MaxPooling1D(pool_size=2))"""
              
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5)) 
    
    model.add(Dense(3,activation='sigmoid') )

    if weights_path:
        model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print("model created")
    return model

def pred(mod,x,locat_x,locat_y,img):
   
    pred = mod.predict(x)
    #print(pred[0].round()==Y[0])
    #print(pred[0])
    #print(pred)

    #print(pred.round == Y )
    
    t = 0 
    f = 0
    cca,hcc,other = 0,0,0
    pred = np.float64(pred)
    #print(max(pred[0]))
    pixel =10
        
    #print("%.10f , %.10f , %.10f"%(pred[i][0],pred[i][1],pred[i][2]))
    if (pred[0][2] == max(pred[0])):            
        cca+=1
        for i in range(0,pixel):
            for j in range(0,pixel):
                
                img[locat_x+i][locat_y+j][0] = 255
        return "c"
    elif (pred[0][1] == max(pred[0])):
            
        hcc+=1
        for i in range(0,pixel):
            for j in range(0,pixel):
                img[locat_x+i][locat_y+j][1] = 255
        return "h"
    elif(pred[0][0] == max(pred[0])):
            
        other+=1
        for i in range(0,pixel):
            for j in range(0,pixel):
                img[locat_x+i][locat_y+j][2] = 255
        return "o"
    
        
    
model = None
def by_RGB(path):

    progress['value'] = 0
    root.update_idletasks()
    model = create_Model_RGB(None)
    thisPath = os.getcwd()
    thisPath = thisPath.replace('\\' , '/' )
    #print(thisPath)
    tp ='C:/Users/Punchill/Desktop/Pro2'
    checkpoint_path = thisPath+"/model_RGB_filter_64_1.ckpt"
    model.load_weights(checkpoint_path)
    imgslide = dowload_test_data(path)
    #print(imgslide[0])
    data = np.asarray(CSV_GLCM_RGB(imgslide))
    #print(data.shape)
    X = data[:,1:len(data)-1]
    #print(X.shape)
    #print(X[0])
    X = np.float64(X)
    pre = 0
    pre0 = []
    pre1 = []
    pre2 = []
    
    lx = []
    ly = []    
    location = np.asarray(data[:,0])
    #print(location)
    for j in location:
        split = (j.split('-'))
        split = split[3]
        split = split.split('.')
        split = split[0]
        split = split.split('_')
        locat_x = int(split[0])
        locat_y = int(split[1])
        lx.append(locat_x)
        ly.append(locat_y)
        progress['value'] += (10/len(location))
        root.update_idletasks()
        
    #print(max(lx))
    #print(max(ly))
    img = 0
    lp =0
    t=5
    PX = []
    PY = []
    img = np.zeros((max(lx)*10,max(ly)*10,3), dtype = np.uint8)
    start_P = time.time()
    print("start predict"+str(time.time()- start_P))
    for k,x,y in zip(X,lx,ly):
        lp+=1 
        k = k.reshape(1,X.shape[1], 1)
            
            #print(k)
        pre = (pred(model,k,(x-1)*10,(y-1)*10,img))
        if pre == "o":
            pre0.append(pre)
        elif pre == "h":
            pre1.append(pre)
        elif pre == "c":
            pre2.append(pre)
        if (time.time() - start_P > t and time.time() - start_P < t+1):
            PY.append(lp)
            PX.append(time.time() - start_P)
            t+=5
            #print(pre)
        progress['value'] += (80/len(ly))
        root.update_idletasks()
        
    print("finish predict : "+str(time.time() - start_P))
            #print(pre)
    
    #print("CCA _ HCC _ Other")    
    #print(len(pre2),len(pre1),len(pre0))
            #cv2.imshow("0",img)
    #cv2.imshow("img",img)
    p = [len(pre2),len(pre1),len(pre0)]
    
    r,c ,d= img.shape
    #print(r,c)
    for i in range(0,r):
        for j in range(0,c):
            if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
        progress['value'] += (10/r)
        root.update_idletasks()
                
    cv2.imwrite("img.png", img)
    s = sum(p)
    plt.plot(PX,PY , 'bo-')
    ro = 0
    for x,y in zip(PX,PY):

        label = "{:.2f}".format(y)
       
        plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
    #plt.show()
    patch = "CCA : "+str(p[0])+" \t|\t HCC : "+str(p[1])+" \t|\t Other : "+str(p[2])
    percent = ("CCA : %.2f %% \t|\t HCC : %.2f %% \t|\t Other : %.2f %%"%(p[0]*100/s,p[1]*100/s,p[2]*100/s))
    return patch,percent
    


def by_Greyscale(path):
    
    progress['value'] = 0
    root.update_idletasks()
    model = create_Model_Greyscale(None)
    thisPath = os.getcwd()
    thisPath = thisPath.replace('\\' , '/' )
    #print(thisPath)
    tp ='C:/Users/Punchill/Desktop/Pro2'
    checkpoint_path = thisPath+'/model_filter_64_1.ckpt'
    start_model = time.time()
    print(start_model)
    model.load_weights(checkpoint_path)
    print("Model Load : "+str(time.time() - start_model))
    imgslide = dowload_test_data(path)
    #print(imgslide[0])
    data = np.asarray(CSV_GLCM_grey(imgslide))
    #print(data.shape)
    X = data[:,1:len(data)-1]
    #print(X.shape)
    #print(X[0])
    X = np.float64(X)
    pre = 0
    pre0 = []
    pre1 = []
    pre2 = []
    
    lx = []
    ly = []    
    location = np.asarray(data[:,0])
    #print(location)
    for j in location:
        split = (j.split('-'))
        split = split[3]
        split = split.split('.')
        split = split[0]
        split = split.split('_')
        locat_x = int(split[0])
        locat_y = int(split[1])
        lx.append(locat_x)
        ly.append(locat_y)
        progress['value'] += (10/len(location))
        root.update_idletasks()
        
    #print(max(lx))
    #print(max(ly))
    img = 0
    img = np.zeros((max(lx)*10,max(ly)*10,3), dtype = np.uint8)
    t = 5
    
    lp =0
    PX = []
    PY = []
    start_P = time.time()
    print("start predict"+str(time.time()- start_P))
    for k,x,y in zip(X,lx,ly):
            
        k = k.reshape(1,X.shape[1], 1)
            
            #print(k)
        pre = (pred(model,k,(x-1)*10,(y-1)*10,img))
        if pre == "o":
            pre0.append(pre)
        elif pre == "h":
            pre1.append(pre)
        elif pre == "c":
            pre2.append(pre)
        lp +=1
        if (time.time() - start_P > t and time.time() - start_P < t+1):
            PY.append(lp)
            PX.append(time.time() - start_P)
            t+=5
            #print(pre)
        progress['value'] += (80/len(ly))
        root.update_idletasks()
    print("finish predict : "+str(time.time() - start_P))
    #print("CCA _ HCC _ Other")    
    #print(len(pre2),len(pre1),len(pre0))
            #cv2.imshow("0",img)
    #cv2.imshow("img",img)
    p = [len(pre2),len(pre1),len(pre0)]
    
    
    r,c ,d= img.shape
    #print(r,c)
    for i in range(0,r):
        for j in range(0,c):
            if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
        progress['value'] += (10/r)
        root.update_idletasks()
        
    cv2.imwrite("img.png", img)
    s = sum(p)
    print("FInish all : "+str(time.time()-start_model))
    patch = "CCA : "+str(p[0])+" \t|\t HCC : "+str(p[1])+" \t|\t Other : "+str(p[2])
    percent = ("CCA : %.2f %% \t|\t HCC : %.2f %% \t|\t Other : %.2f %%"%(p[0]*100/s,p[1]*100/s,p[2]*100/s))
    plt.plot(PX,PY , 'bo-')
    ro = 0
    for x,y in zip(PX,PY):

        label = "{:.2f}".format(y)
       
        plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
    #plt.show()
    return patch,percent




def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=False, num_classes=3):
        super(GoogLeNet, self).__init__()
        assert aux_logits == False or aux_logits == False
        self.aux_logits = aux_logits

        # Write in_channels, etc, all explicit in self.conv1, rest will write to
        # make everything as compact as possible, kernel_size=3 instead of (3,3)
        self.conv1 = conv_block(
            in_channels=1,
            out_channels=64,
            kernel_size=(249, 249)
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, 3)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        # if self.aux_logits and self.training:
        #     aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        # if self.aux_logits and self.training:
        #     aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        # if self.aux_logits and self.training:
        #     return aux1, aux2, x
        # else:
        #     return x
        return x

class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
def get_test_data(batch_size):
        im_size = 249
        data_path = 'D:/Data_inception/Test/Other/1/'
        test_dataset = torchvision.datasets.ImageFolder(
          data_path,
          transform = transforms.Compose([
                      transforms.Grayscale(num_output_channels=1),
                      transforms.Resize(size=(im_size, im_size)),
                      transforms.ToTensor(),
                      #transforms.RandomCrop(200),
                      transforms.Normalize((0.485), (0.299))
                      
                      ]) 
        )
        test_loader = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=batch_size,
          #train=False,
          #collate_fn=default_collate,
          shuffle=False
        )
        return test_loader
def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        print(checkpoint)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
                parameter.requires_grad = False
        model.eval()
        return model

loader = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(),transforms.Normalize((0.485),(0.299))])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    #print(image)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU
def get_images(namePath):
    print(dr)
    datas = []
    i= 0
    namePath += "/"
    print(namePath)

    #namePath = 'D:/Dataset_GLCM/Dataset/Test/HCC/S63-4504-1E/'
    for (d,c,path) in os.walk (namePath):
            
        for image in path:
            i+=1
            data = namePath+image
            datas.append(data)
                #print(image)
            #for (d,c,path) in os.walk(image):
    
    #print(len(datas))
    #print(datas[0])
    return datas







        
def inception(path):
    start = time.time()
    print("start running : "+str(time.time() - start))
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    time_P = []
    data_P = []
    


    p = [0,0,0]
    pl = ['CCA' , 'HCC' , 'Other']


    model = GoogLeNet()
    model.to(device)
    model = load_checkpoint('0_422.pth')
    print("get_model : "+str(time.time() - start))
    start_image = time.time()
    datas = get_images(path)
    print(len(datas)) 
    lx = []
    ly = []
    pre = []
    print("get_image_Path : "+str(time.time() - start_image))
    t = 1
    print(t)
    progress['value'] = 0
    root.update_idletasks()
    start_P = time.time()
    with torch.no_grad():
        for data in datas:
            split = data.split('/')
            location = (split[-1])
            image = image_loader(data)
            
            image = image.to(device)
            data = torch.FloatTensor(image)
               
            
            
            split = (location.split('-'))
            split = split[3]
            split = split.split('.')
            split = split[0]
            split = split.split('_')
            locat_x = int(split[0])
            locat_y = int(split[1])
            #print(locat_x,locat_y)
            lx.append((locat_x-1)*10)
            ly.append((locat_y-1)*10)
                          

            
            data = data.to(device)
                  

            output = model(data)
           
            _, predicted = torch.max(output, 1)
            
            pred = torch.max(output, 1)[1].data.squeeze()
            
           
            p[predicted] +=1
            pre.append(predicted)
            #if (len(pre) % (len(datas) // 5) == 0):
            if time.time() - start_P > t and time.time() - start_P < t+1:
                data_P.append(len(pre))
                time_P.append(time.time() - start_P)
                t+=1
            progress['value'] += (80/len(datas))
            root.update_idletasks()
    print("Predicted : "+str(time.time() - start_P))
    #print(str(time_P /len(data)))
    img = 0
    img = np.zeros((max(lx)+10,max(ly)+10,3), dtype = np.uint8)
    pixel = 10
    
    for i in range(0,len(pre)):
            for j in range(0,pixel):
                    for k in range(0,pixel):
                            #print(lx[i] , ly[i])
                            img[lx[i]+j,ly[i]+k,pre[i]] = 255
            progress['value'] += (10/len(pre))
            root.update_idletasks()
    #cv2.imshow('img',img)
    s = sum(p)
    patch = "CCA : "+str(p[0])+" \t|\t HCC : "+str(p[1])+" \t|\t Other : "+str(p[2])
    #percent = "CCA : "+str(p[0]*100/s)+"% \t| HCC : "+str(p[1]*100/s)+"% \t| Other : "+str(p[2]*100/s)+"%"
    percent = ("CCA : %.2f %% \t|\t HCC : %.2f %% \t|\t Other : %.2f %%"%(p[0]*100/s,p[1]*100/s,p[2]*100/s))
    #for i in range(0,3):
     #       if p[i] == max(p):
      #              print("Most Predicted : "+str(pl[i]))
    r,c ,d= img.shape
  
    for i in range(0,r):
        for j in range(0,c):
            if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
        progress['value'] += (10/r)
        root.update_idletasks()
    cv2.imwrite("img.png", img)
    
    
    plt.plot(time_P,data_P , 'bo-')
    ro = 0
    for x,y in zip(time_P,data_P):

        label = "{:.2f}".format(y)
       
        plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
    #plt.show()
    print('finish all'+str(time.time()-start))
    return patch,percent
def runPro(Event):
    patch = ""
    percent = ""

    if cFunc.get() == "Inception":
        
        patch , percent  = inception(dr)
        
        

    elif cFunc.get() == "GLCM Gray scale":
        patch , percent = by_Greyscale(dr)

    elif cFunc.get() == "GLCM RGB":
        patch , percent = by_RGB(dr)
    

    photo = ImageTk.PhotoImage(Image.open('img.png'))
    cLabel3=Label(top_frame, width=130, height=25)
    cLabel3.configure(image=photo)
    cLabel3.image = photo
    cLabel3.grid(pady=10, row=5, column=1, columnspan=4, sticky="NESW")
    
    cEntry3['state'] = 'normal'
    cEntry3.delete(0,"end")
    cEntry3.insert(0, patch)

    cEntry4['state'] = 'normal'
    cEntry4.delete(0,"end")
    cEntry4.insert(0, percent)
    print(patch)
    print(percent)

def getFile(Event):
    global filename,listbox,dr
    dr = filedialog.askdirectory(initialdir=os.getcwd,title="Please select a folder:")
    #print(Mfile)
    cEntry1['state'] = 'normal'
    cEntry1.delete(0,"end")
    cEntry1.insert(0, dr)
    

def getLink(Event):
    cEntry1['state'] = 'disabled'
    cEntry2['state'] = 'normal'
    
        
OptionList = ["Inception","GLCM Gray scale","GLCM RGB"]
root=Tk()
panel=None
txt=None
#root.resizable(width=TRUE, height=TRUE)
root.geometry('{}x{}'.format(1600, 760))
#root.attributes('-fullscreen', True)
toggle_geom='200x200+0+0'
#root.geometry("{0}x{1}+0+0".format(
    #root.winfo_screenwidth()-3, root.winfo_screenheight()-3))
root.bind('<Escape>',toggle_geom)
root.configure(background='sea green')
root.title('การวิเคราะห์ภาพทางการแพทย์สำหรับโรคมะเร็งท่อน้ำดีและมะเร็งตับ')

top_frame = Frame(root, bg='dark sea green')
center = Frame(root, bg='sea green')
#sub_center = Frame()

menubar=Menu(root)
fileMenu=Menu(menubar,tearoff=0)
fileMenu.add_command(label="Exit",command=root.destroy)
menubar.add_cascade(label="File",menu=fileMenu)
root.config(menu=menubar)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
top_frame.grid(row=0, sticky="EW", padx=10, pady=10)
center.grid(row=1, sticky="EW")


model_label = Label(top_frame, text='การวิเคราะห์ภาพทางการแพทย์สำหรับโรคมะเร็งท่อน้ำดีและมะเร็งตับ', bg='dark sea green')
model_label.config(font=('tahoma', 20, 'bold'))
model_label.grid(row=0, column=0, columnspan=6, sticky="WENS")

eLabel1 = Label(top_frame, text='\t', bg='dark sea green')
eLabel1.grid(row=1, column=0, columnspan=4, sticky="EN")

#row2
cFile=Button(top_frame, text='เลือกไฟล์',bg='forest green',fg='white',width=10)
cFile.config(font=('helvetica', 16, 'bold'))
cFile.bind('<Button-1>', getFile)
cFile.grid(padx=15, pady=5, row=2, column=0, sticky="N")

cEntry1=Entry(top_frame, width=76)
cEntry1.config(font=('helvetica', 14, 'bold'))
cEntry1.grid(pady=10, row=2, column=1, columnspan=4, sticky="NSW")

cRun=Button(top_frame, text='Run',bg='forest green',fg='white',width=10)
cRun.config(font=('helvetica', 14, 'bold'))
cRun.bind('<Button-1>', runPro)
cRun.grid(padx=15, pady=5, row=2, column=4, sticky="N")

#row3
progress = ttk.Progressbar(top_frame, orient = HORIZONTAL, length = 840, mode = 'determinate', value = 0)
progress.grid(pady=5, row=3, column=1, sticky="SW")

#row4
cLabel2 = Label(top_frame, text='วิธีการ : ', bg='dark sea green', width=10)
cLabel2.config(font=('helvetica', 16, 'bold'))
cLabel2.grid(padx=15, pady=5, row=4, column=0, sticky="S")

variable = tk.StringVar()
variable.set(OptionList[0])

cFunc = ttk.Combobox(top_frame, textvariable=variable, value = OptionList)
cFunc.config(font=('helvetica', 14, 'bold'), width=30)
cFunc.grid(pady=5, row=4, column=1, sticky="SW")



#row5


cLabel3=Label(top_frame, width=201, height=25)
cLabel3.grid(padx=24, pady=10, row=5, column=0, columnspan=8, sticky="NESW")

#row6
cLabel4 = Label(top_frame, text='Predicted per patch : ', bg='dark sea green', width=20)
cLabel4.config(font=('helvetica', 16, 'bold'))
cLabel4.grid(padx=15, pady=5, row=6, column=0, sticky="N")

cEntry3=Entry(top_frame, width=76 , state='disabled')
cEntry3.config(font=('helvetica', 20, 'bold'))
cEntry3.grid(pady=10, row=6, column=1, columnspan=4, sticky="NSW")

#row7
cLabel5 = Label(top_frame, text='Predicted percent : ', bg='dark sea green', width=20)
cLabel5.config(font=('helvetica', 16, 'bold'))
cLabel5.grid(padx=15, pady=5, row=7, column=0, sticky="N")

cEntry4=Entry(top_frame, width=76 , state='disabled')
cEntry4.config(font=('helvetica', 20, 'bold'))
cEntry4.grid(pady=10, row=7, column=1, columnspan=4, sticky="NSW")

#row8
cLabel6 = Label(top_frame, text='CCA-Blue', bg='blue')
cLabel6.config(font=('helvetica', 16, 'bold'))
cLabel6.grid(padx=15, pady=5, row=8, column=1, sticky="NE")

cLabel7 = Label(top_frame, text='HCC-Green', bg='green')
cLabel7.config(font=('helvetica', 16, 'bold'))
cLabel7.grid(padx=15, pady=5, row=8, column=2, sticky="N")

cLabel8 = Label(top_frame, text='Other-Red', bg='red')
cLabel8.config(font=('helvetica', 16, 'bold'))
cLabel8.grid(padx=15, pady=5, row=8, column=3, sticky="N")
root.mainloop()
