# -*- coding: utf-8 -*-
'''
Face recognition.
@author     Matthias Moulin
@version    1.0

PS.: note that http://stackoverflow.com/questions/22885100/principal-component-analysis-in-python-analytical-mistake/22885447?noredirect=1#comment34945020_22885447
is my account and my question and that the answer is based on my code (which was only available for some hours via dropbox)
'''

import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch


def create_database(directory, show = True):
    '''
    Process all images in the given directory.
    Every image is 
    cropped to the detected face, 
    resized to 100x100 and 
    saved in another directory (orignal directory name + "2").
    @param directory:    directory to process
    @param show:         bool, show all intermediate results
    '''
    #load a pre-trained classifier
    #
    #cv2.CascadeClassifier:
    #
    #First, a classifier (namely a cascade of boosted classifiers working with haar-like features) is trained with a few hundred
    #sample views of a particular object (i.e., a face or a car), called positive examples, that are scaled to the same size 
    #(say, 20x20), and negative examples - arbitrary images of the same size.
    #
    #After a classifier is trained, it can be applied to a region of interest (of the same size as used during the training) 
    #in an input image. The classifier outputs a “1” if the region is likely to show the object (i.e., face/car), and “0” otherwise. 
    #To search for the object in the whole image one can move the search window across the image and check every location using the 
    #classifier. The classifier is designed so that it can be easily “resized” in order to be able to find the objects of interest
    #at different sizes, which is more efficient than resizing the image itself. So, to find an object of an unknown size in the
    #image the scan procedure should be done several times at different scales.
    #
    #The word “cascade” in the classifier name means that the resultant classifier consists of several simpler classifiers (stages)
    #that are applied subsequently to a region of interest until at some stage the candidate is rejected or all the stages are passed.
    #The word “boosted” means that the classifiers at every stage of the cascade are complex themselves and they are built out of
    #basic classifiers using one of four different boosting techniques (weighted voting).
    cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    
    #loop through all files
    #
    #os.listdir:
    #
    #The method listdir() returns a list containing the names of the entries in the directory given by path.
    #The list is in arbitrary order. It does not include the special entries '.' and '..' even if they are present in the directory.
    #
    #fnmatch.filter:
    #
    #Return the subset of the list of names that match pattern.
    for filename in fnmatch.filter(os.listdir(directory),'*.jpg'):
        file_in = directory+"/"+filename
        file_out= directory+"2/"+filename
        img = cv2.imread(file_in)
        if show:
            cv2.imshow('img',img)
            cv2.waitKey(0)
        #do face detection    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Image Histogram:
        #
        #It is a graphical representation of the intensity distribution of an image.
        #It quantifies the number of pixels for each intensity value considered.
        #
        #Histogram Equalization:
        #
        #It is a method that improves the contrast in an image, in order to stretch out the intensity range.
        #Equalization implies mapping one distribution (the given histogram) to another distribution 
        #(a wider and more uniform distribution of intensity values) so the intensity values are spreaded over the whole range.
        gray = cv2.equalizeHist(gray)
        #detectMultiScale:
        #
        #Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
        #     scaleFactor     – Parameter specifying how much the image size is reduced at each image scale.
        #     minNeighbors    – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        #     flags           – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. 
        #     minSize         – Minimum possible object size. Objects smaller than that are ignored.
        #     maxSize         – Maximum possible object size. Objects larger than that are ignored.
        rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
        #result is an array of x coordinate, y coordinate, weight, height for each rectangle
        rects[:,2:] += rects[:,:2]
        #result is an array of x coordinate, y coordinate, x + weight, y + height for each rectangle (opposite corners)
        rects = rects[0,:]
        #result, just one single rectangle is considered
        #show detected result
        if show:
            vis = img.copy()
            cv2.rectangle(vis, (rects[0], rects[1]), (rects[2], rects[3]), (0, 255, 0), 2)
            cv2.imshow('img', vis)
            cv2.waitKey(0)
        
        #crop image to the rectangle and resample it to 100x100 pixels
        result = cv2.resize(img[rects[1]:rects[3], rects[0]:rects[2], :], (100,100))
        
        print(file_out)
        #show result
        if show:
            cv2.imshow('img', result)
            cv2.waitKey(0)            
        #save the face in a second directory
        cv2.imwrite(file_out, result)
    cv2.destroyAllWindows()

def createX(directory, nbDim=10000):
    '''
    Create an array that contains all the images in directory.
    @param directory:    directory to process
    @param nbDim:        the number of dimensions of each sample
    @return np.array, shape=(nb images in directory, nb pixels in image)
    '''
    #os.listdir:
    #
    #The method listdir() returns a list containing the names of the entries in the directory given by path.
    #The list is in arbitrary order. It does not include the special entries '.' and '..' even if they are present in the directory.
    #
    #fnmatch.filter:
    #
    #Return the subset of the list of names that match pattern.
    filenames = fnmatch.filter(os.listdir(directory),'*.jpg')
    nbImages = len(filenames)
    X = np.zeros((nbImages, nbDim))
    for i,filename in enumerate(filenames):
        file_in = directory+"/"+filename
        img = cv2.imread(file_in)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        X[i,:] = gray.flatten()
    return X


def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    return np.dot(W.T, (X-mu))

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    return (np.dot(W, Y) + mu)

def pca(X, nb_components=0):
    '''
    Do a PCA (Principal Component Analysis) on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n
    
    #Turn a set of possibly correlated variables into a smaller set of uncorrelated variables.
    #The idea is, that a high-dimensional dataset is often described by correlated variables and
    #therefore only a few meaningful dimensions account for most of the information.
    #The PCA method finds the directions with the greatest variance in the data, called principal components.
    
    MU = X.mean(axis=0)
    for i in range(n):
        X[i,:] -= MU
    
    S = (np.dot(X, X.T) / float(n))
    #Hermitian (or symmetric) matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    
    #And about the negative eigenvalues, it is just a matter of eigh.
    #As eigenvalues shows the variance in a direction, we care about absolute
    #value but if we change a sign, we also have to change the "direcction" (eigenvector).
    #You can make this multiplying negative eigenvalues and their corresponding eigenvectors with -1.0
    s = np.where(eigenvalues < 0)
    eigenvalues[s] = eigenvalues[s] * -1.0
    eigenvectors[:,s] = eigenvectors[:,s] * -1.0

    #The nb_components largest eigenvalues and eigenvectors of the covariance matrix
    indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indexes][0:nb_components]
    eigenvectors = eigenvectors[:,indexes][:,0:nb_components]

    eigenvectors = np.dot(X.T, eigenvectors)
    for i in range(nb_components):
        eigenvectors[:,i] = normalize_vector(eigenvectors[:,i])
    
    return (eigenvalues, eigenvectors, MU)

def normalize_vector(v):
    '''
    Normalize the given vector.
    '''
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)

if __name__ == '__main__':
    #create database of normalized images
    for directory in ["data/arnold", "data/barack"]:
        create_database(directory, show = False)
    
    show = True
    
    # create big X arrays for arnold and barack
    Xa = createX("data/arnold2")
    Xb = createX("data/barack2")
            
    #Take one part of the images for the training set, the rest for testing
    nbTrain = 6
    
    #vstack:
    #
    #Stack arrays in sequence vertically (row wise).
    #Take a sequence of arrays and stack them vertically to make a single array.
    Xtest = np.vstack((Xa[nbTrain:,:],Xb[nbTrain:,:]))
    Ctest = ["arnold"]*(Xa.shape[0]-nbTrain) + ["barack"]*(Xb.shape[0]-nbTrain)
    #E.g.: ['arnold', 'arnold', 'barack', 'barack', 'barack', 'barack']
    Xa = Xa[0:nbTrain,:]
    Xb = Xb[0:nbTrain,:]

    #do pca
    #4 or 5 components will do the job
    [eigenvaluesa, eigenvectorsa, mua] = pca(Xa, nb_components=5)
    [eigenvaluesb, eigenvectorsb, mub] = pca(Xb, nb_components=5)
    
    #visualize
    #
    #hstack:
    #
    #Stack arrays in sequence horizontally (column wise).
    cv2.imshow('img',np.hstack( (mua.reshape(100,100),
                                 normalize(eigenvectorsa[:,0].reshape(100,100)),
                                 normalize(eigenvectorsa[:,1].reshape(100,100)),
                                 normalize(eigenvectorsa[:,2].reshape(100,100)),
                                 normalize(eigenvectorsa[:,3].reshape(100,100)),
                                 normalize(eigenvectorsa[:,4].reshape(100,100))),
                                 #normalize(eigenvectorsa[:,5].reshape(100,100))),
                               ).astype(np.uint8))
    cv2.waitKey(0) 
    cv2.imshow('img',np.hstack( (mub.reshape(100,100),
                                 normalize(eigenvectorsb[:,0].reshape(100,100)),
                                 normalize(eigenvectorsb[:,1].reshape(100,100)),
                                 normalize(eigenvectorsa[:,2].reshape(100,100)),
                                 normalize(eigenvectorsb[:,3].reshape(100,100)),
                                 normalize(eigenvectorsb[:,4].reshape(100,100))),
                                 #normalize(eigenvectorsb[:,5].reshape(100,100))),
                               ).astype(np.uint8))
    cv2.waitKey(0) 
           
    nbCorrect = 0
    for i in range(Xtest.shape[0]):
        X = Xtest[i,:]
       
        #project image i on the subspace of arnold and barack
        Ya = project(eigenvectorsa, X, mua )
        Xa = reconstruct(eigenvectorsa, Ya, mua)
        Yb = project(eigenvectorsb, X, mub )
        Xb = reconstruct(eigenvectorsb, Yb, mub)
        
        if show:
            #show reconstructed images
            cv2.imshow('img',np.hstack((X.reshape(100,100),
                                        np.clip(Xa.reshape(100,100), 0, 255),
                                        np.clip(Xb.reshape(100,100), 0, 255)) ).astype(np.uint8) )
            cv2.waitKey(0)   

        #classify i
        print "Image nb "+str(i)
        print "SSD image & model Arnold:"+str(np.linalg.norm(Xa-Xtest[i,:]) )
        print "SSD image & model Barack:"+str(np.linalg.norm(Xb-Xtest[i,:]))
        
        if np.linalg.norm(Xa-Xtest[i,:]) < np.linalg.norm(Xb-Xtest[i,:]):
            bestC = "arnold"
        else:
            bestC = "barack"
        print str(i)+":"+str(bestC)
        print "---"
        
        if bestC == Ctest[i]:
           nbCorrect+=1
    
    #Print final result
    print str(nbCorrect)+"/"+str(len(Ctest))
    
