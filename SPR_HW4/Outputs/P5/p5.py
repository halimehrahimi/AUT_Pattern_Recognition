
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random


# ## Reading the Data

# In[2]:


data = pd.read_json("F:/Uni/991/Pattern/SPR_HW4/inputs/P5/tweets.json", lines=True)
tw = data[['id', 'text']]


# In[3]:


#data.head()


# In[4]:


#tw.head()


# ## Functions

# In[5]:


#computing jaccard similarity, I use sets because the order is not important in them
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


# In[6]:


#computing 1-j
def jaccard(list1,list2):
    j = jaccard_similarity(list1, list2)
    return 1-j


# In[7]:


# initializing the centroids, if rand=True we'll have kmeans++ else we will use the given centroids
def init_centroid(rand, k, X):
    centroids=list()
    if rand==False:
        #using the given cenroids
        c = open('F:/Uni/991/Pattern/SPR_HW4/inputs/P5/initial_centroids.txt', 'r')
        for line in c:
            last = line
            centroids.append(line[:-2])
        centroids=centroids[:-1]
        centroids.append(last)
    else:
        #first centroid is chosen randomly
        centroids = list()
        choice = random.sample(list(X['id']), 1)[0]
        centroids.append(choice)
        #I drop it so it wouldn't get calculated again
        X_d = X.drop(index=np.where(X['id']==choice)[0][0]).reset_index(drop=True)
        #Now we're finding the distance between points
        #The farthest point from the chosen centroids up until then are chosen as the centroid
        for c_id in range(1, k): 
            dist = list()
            for i in range(X_d.shape[0]): 
                d = list()
                point = list(X_d.loc[i, 'text'].split())
                for j in range(len(centroids)):
                    d.append(jaccard(point, list(X.iloc[np.where(X.id==centroids[j])[0][0], 1].split())))
                dist.append(min(d)) #the minimum distance a point has from any of the centroids 
            next_centroid = X_d.loc[np.argmax(dist), 'id'] #choosing the farthest point
            X_d = X_d.drop(index=np.where(X_d['id']==next_centroid)[0][0]).reset_index(drop=True)
            #dropping it so it doesn't get calculated again
            centroids.append(next_centroid) #adding the centroid
    return centroids


# In[8]:


#assigning the closest centroid to each point
def find_closest_centroids(cent, X):
    assigned_centroid = list()
    #calculating the distance
    for i in range(0,X.shape[0]):
        distance=list()
        s=list(X.iloc[i,1].split())
        for j in cent:
            distance.append(jaccard(s, list(X.iloc[np.where(X.id==int(j))[0][0],1].split())))
        assigned_centroid.append(cent[np.argmin(distance)])
    new_df = pd.concat([pd.DataFrame(X), pd.DataFrame(assigned_centroid, columns=['cluster'])],axis=1) #cluster assignment
    return new_df


# In[9]:


#updating the centroids
def calc_centroids(centroids,new_df):
    new_centroids = list()
    for c in centroids:
        #get all the points in a cluster
        current_cluster = new_df.iloc[np.where(new_df['cluster']==c)[0]].reset_index(drop=True)
        distance_avg = list()
        #calculating the average of the pairwise distance
        for i in current_cluster['text']:
            distance = list()
            for j in current_cluster['text']:
                distance.append(jaccard(list(i.split()),list(j.split())))
            distance_avg.append(np.mean(distance)) #finding the point that has the minimum average distance to other points in the cluster
        new_centroids.append(current_cluster.loc[np.argmin(distance_avg),'id']) #new centroids
    
    return new_centroids


# In[10]:


#creating the model
def kmeans(k,rand,X):
    #initialization
    centroids = init_centroid(rand, k, X)
    t = 0
    print('Initial Centroids:\n', centroids)
    #the loop will stop whenever there is no update, this is done by compring the previous and the current centroids list
    while t==0:
        prev_centroids = centroids.copy()
        #Assigning Clusters
        new_df = find_closest_centroids(centroids, X)
        #print(labels)
        #Updating Centroids
        centroids = calc_centroids(centroids,new_df)
        print('---------------------')
        print('New Centroids:')
        print(centroids)
        if prev_centroids == centroids:
            t=1

    return centroids,new_df


# ## a.

# In[11]:


print('Part a')
rand = False
k = 25
print('K-means with given Initial Centroids:')
centroids1,new_df1 = kmeans(k,rand,tw)


# In[12]:


f = open("F:/Uni/991/Pattern/SPR_HW4/inputs/P5/Clusters_for_Given_Centroids.txt", "w")
for c in centroids1:
    f.write(str(c)+':\n')
    for d in new_df1.loc[np.where(new_df1['cluster']==c)[0],'id']:
        f.write(str(d) + ',')
    f.write('\n')
f.close()


# ## b.

# In[13]:


print('Part b')
rand = True
k = 25
print('K-means++:')
centroids2,new_df2 = kmeans(k,rand,tw)


# In[14]:


f = open("F:/Uni/991/Pattern/SPR_HW4/inputs/P5/Clusters_for_Kmeans++.txt", "w")
for c in centroids2:
    f.write(str(c)+':\n')
    for d in new_df2.loc[np.where(new_df2['cluster']==c)[0],'id']:
        f.write(str(d) + ',')
    f.write('\n')
f.close()


# ## extra

# In[15]:


#similarity between the centroids from part a and b
list1 = centroids1
list2 = centroids2
print('Similarity between the chosen centroids in part a and b:\n', jaccard_similarity(list1, list2))

