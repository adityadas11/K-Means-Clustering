#!/usr/bin/env python
# coding: utf-8

# In[32]:


import scipy.io
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from random import randint


# In[33]:


#Loading the sample.mat file
numpyFile = loadmat("AllSamples.mat")


# In[34]:


#Extracting the samples from the file
samples = numpyFile["AllSamples"]


# In[43]:


class KMeans:
    def __init__(self, k=2, tol=0.001, maxIter=1000, strategy=1):
        # Number of clusters
        self.k = k
        # Tolerance percentage to stop the algorithm
        self.tol = tol
        # Global objective function value for each K
        self.globalObjectiveFunction = 0
        # Maximum iterations willing to do
        self.maxIter = maxIter
        # assigning the strategy as per the parameter
        self.strategy = strategy
        
        
        # Strategy 1
    def randomCentroid(self, data):
        # Selecting random k centroids from the data sample
        for i in range(self.k):
            # Selecting a random index
            randomNumber = randint(i, len(data)-1)
            # Swapping random index number with i th index
            temp = data[randomNumber].copy()
            data[randomNumber] = data[i]
            data[i] = temp
            
            # Strategy 2
    def randomCentroidMaxDistance(self, data):
        for i in range(self.k):
            if i == 0:
                # Selecting the first centroid randomly
                randomNumber = randint(i, len(data)-1)
                temp = data[randomNumber].copy()
                data[randomNumber] = data[i]
                data[i] = temp
            else:
                # Selecting maximum distance centroid from all the selected centroid
                maxDistanceFromCentroid = i
                globalSum = 0
                # Calculating the average distance of point from all the selected centroids
                for pos in range(i, len(data)):
                    localSum = 0;
                    for q in range(0, i):
                        localSum = localSum + np.linalg.norm(data[pos]-data[q]) 
                        # Updating point giving maximum distance to all selected centroids
                        if localSum > globalSum:
                            globalSum = localSum
                            maxDistanceFromCentroid = pos
                # Swapping positions
                temp = data[maxDistanceFromCentroid].copy()
                data[maxDistanceFromCentroid] = data[i]
                data[i] = temp
                
      
    #fit function to fit the datapoints to clusters
    def fit(self, data):
        # Choosing centroids according to chosen strategy
        if self.strategy == 1:
            self.randomCentroid(data)
        else:
            self.randomCentroidMaxDistance(data)
        # a dictionary of current centroids
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
        # Local object function to track value for each iteration
        localObjectiveFunc = {}
        for i in range(self.maxIter):
            # Classification dictionary keeping track of points under each centroid
            self.classifications = {}
            for m in range(self.k):
                # Initializing empty list for each classification
                self.classifications[m] = []
            # For each point in dataset, calculating distance to every chosen centroid and updating them
            # the classification list of the centroid
            for point in data:
                distances = [np.linalg.norm(point-self.centroids[centroidPos]) for centroidPos in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(point)
            # Creating a dictionary to keep track of previous set of centroids
            prevCentroids = dict(self.centroids)
            localSum = 0
            # Calculating local objective function value for iteration
            for classification in self.classifications:
                sumOfSquaredDistances = 0
                for point in self.classifications[classification]:
                    distance = np.linalg.norm(point-self.centroids[classification])
                    sumOfSquaredDistances = sumOfSquaredDistances + (distance * distance)
                localSum = localSum + sumOfSquaredDistances
            # Updating local objective function value
            localObjectiveFunc[i] = localSum
            # Updating global objective function value
            self.globalObjectiveFunction = localSum
            # Updating centroid after iteration
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            # Assuming the centroids are not fluctuating significantly anymore and the solution has converged.
            optimized = True
            # Checking if every centroid movement is within the threshold range
            for c in self.centroids:
                originalCentroid = prevCentroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-originalCentroid)/originalCentroid*100.0) > self.tol:
                    optimized = False
            # If centroid movement is still over the threshold, then continuing the iteration else breaking 
            if optimized:
                break
        return self.globalObjectiveFunction
    
    
    
    
    def predict(self, data):
        classificationResult = {}
        for point in data:
            # Finding centroid closest to the final centroids and appending it to the dictionary
            distances = [np.linalg.norm(point-self.centroids[centroidPos]) for centroidPos in self.centroids]
            classification = distances.index(min(distances))
            if classificationResult.get(classification) == None:
                classificationResult[classification] = []
            classificationResult[classification].append(point)
        # Scatter plot of all the clustered points and final centroids
        for centroid in classificationResult.keys():
            values = classificationResult.get(centroid)
            x_points = [point[0] for point in values]
            y_points = [point[1] for point in values]
            plt.scatter(x_points, y_points)
        self.plotGraph()
        return classification
    
    
    def plotGraph(self):
        centroids = self.centroids.values()
        for centroid in centroids:
            plt.scatter(centroid[0], centroid[1], marker="o", color="k")
            plt.ylabel("K = " +str(self.k))
            plt.xlabel("Strategy = " +str(self.strategy))
        plt.show()

                


# In[44]:


# Covering all strategies
for i in range(1, 3):
    objectiveFunction = {}
    # Covering all K values
    for _k in range(2, 11):
        x = KMeans(k=_k, strategy=i)
        objectiveFunction[_k] = x.fit(samples)
        x.predict(samples)
    #plotting the K value in the x axis and strategy number in y
    plt.plot(list(objectiveFunction.keys()), list(objectiveFunction.values()))
    plt.xlabel("K")
    plt.ylabel("Strategy " +str(i))
    plt.show()


# In[ ]:





# In[ ]:




