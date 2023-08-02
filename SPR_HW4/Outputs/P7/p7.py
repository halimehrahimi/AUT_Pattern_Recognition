
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.data import loadlocal_mnist
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


x, y = loadlocal_mnist(images_path='F:/Uni/991/Pattern/SPR_HW4/inputs/P7/t10k-images-idx3-ubyte',
                                 labels_path='F:/Uni/991/Pattern/SPR_HW4/inputs/P7/t10k-labels-idx1-ubyte')


# ## a

# In[3]:


#computing principal components
print('Part a')
pca = PCA()
xnew = pca.fit_transform(x)
eigvec = pca.components_
eigval = pca.explained_variance_


# In[4]:


print('Top 20 Eigenvalues:\n', eigval[:20])


# In[5]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    plt.scatter(xnew[y==val][:,0], xnew[y==val][:,1], label=val)
    plt.legend()
    plt.title('Samples Projected onto their First two Principal Components')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.show()


# In[6]:


fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)
for val in np.unique(y):
    ax.scatter(xnew[y==val][:,0], xnew[y==val][:,1], xnew[y==val][:,2], label=val)
    ax.legend()
    plt.title('Samples Projected onto their First three Principal Components')
    ax.set_xlabel('First Feature')
    ax.set_ylabel('Second Feature')
    ax.set_zlabel('Third Feature')
plt.show()


# ## b

# In[7]:


print('Part b')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[8]:


#computing linear discriminants
lda = LDA()
x_lda = lda.fit_transform(x, y)


# In[9]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    points = x_lda[y==val][:,0]
    p = np.zeros(points.shape)
    plt.scatter(points,p, label=val)
    plt.legend()
    plt.title('Samples Projected onto their First Linear Discriminants')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.show()


# In[10]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    plt.scatter(x_lda[y==val][:,0], x_lda[y==val][:,1], label=val)
    plt.legend()
    plt.title('Samples Projected onto their First two Linear Discriminants')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.show()


# ## c

# In[44]:


#4-means with random initialization
print('Part c')
kmeans4 = KMeans(n_clusters=4, init='random', random_state=0).fit(xnew[:,:2])
centroids4 = kmeans4.cluster_centers_
print('Final Centroids for random 4-means:\n', centroids4)


# In[12]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    plt.scatter(xnew[y==val][:,0], xnew[y==val][:,1], label=val)
    plt.legend()
    plt.title('Part c\n4-means Random Initialization')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.scatter(centroids4[:,0], centroids4[:,1], c='black')
plt.show()


# In[45]:


#7-means with random initialization
kmeans7 = KMeans(n_clusters=7, init='random', random_state=0).fit(xnew[:,:2])
centroids7 = kmeans7.cluster_centers_
print('Final Centroids for random 7-means:\n', centroids7)


# In[14]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    plt.scatter(xnew[y==val][:,0], xnew[y==val][:,1], label=val)
    plt.legend()
    plt.title('Part c\n7-means Random Initialization')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.scatter(centroids7[:,0], centroids7[:,1], c='black')
plt.show()


# In[46]:


#10-means with random initialization
kmeans10 = KMeans(n_clusters=10, init='random', random_state=0).fit(xnew[:,:2])
centroids10 = kmeans10.cluster_centers_
print('Final Centroids for random 10-means:\n', centroids10)


# In[16]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    plt.scatter(xnew[y==val][:,0], xnew[y==val][:,1], label=val)
    plt.legend()
    plt.title('Part c\n10-means Random Initialization')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.scatter(centroids10[:,0], centroids10[:,1], c='black')
plt.show()


# ## d

# In[17]:


#centroids initialization for 4-means
print('Part d')
mean41=np.expand_dims(np.concatenate((xnew[y==0,:2], xnew[y==2,:2], xnew[y==4,:2], xnew[y==6,:2]), 0).mean(0),0)
mean42=np.expand_dims(np.concatenate((xnew[y==1,:2], xnew[y==3,:2]), 0).mean(0),0)
mean43=np.expand_dims(np.concatenate((xnew[y==5,:2], xnew[y==7,:2], xnew[y==9,:2]), 0).mean(0),0)
mean44=np.expand_dims(xnew[y==8,:2].mean(0),0)

init_centroids4 = np.concatenate((mean41, mean42, mean43, mean44),0)


# In[18]:


print('Initial Centroids for 4-means:\n', init_centroids4)


# In[19]:


#4-means with initialization
kmeans4 = KMeans(n_clusters=4, init=init_centroids4, random_state=0).fit(xnew[:,:2])
centroids4 = kmeans4.cluster_centers_
print('Final Centroids for 4-means:\n',centroids4)


# In[20]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    plt.scatter(xnew[y==val][:,0], xnew[y==val][:,1], label=val)
    plt.legend()
    plt.title('Part d\n4-means Given Initial Centroids')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.scatter(centroids4[:,0], centroids4[:,1], c='black')
plt.show()


# In[21]:


#centroids initialization for 7-means
mean71=np.expand_dims(np.concatenate((xnew[y==0,:2], xnew[y==2,:2], xnew[y==4,:2]), 0).mean(0),0)
mean72=np.expand_dims(xnew[y==1,:2].mean(0),0)
mean73=np.expand_dims(xnew[y==3,:2].mean(0),0)
mean74=np.expand_dims(xnew[y==5,:2].mean(0),0)
mean75=np.expand_dims(xnew[y==6,:2].mean(0),0)
mean76=np.expand_dims(xnew[y==8,:2].mean(0),0)
mean77=np.expand_dims(np.concatenate((xnew[y==7,:2], xnew[y==9,:2]), 0).mean(0),0)

init_centroids7 = np.concatenate((mean71, mean72, mean73, mean74, mean75, mean76, mean77),0)


# In[22]:


print('Initial Centroids for 7-means:\n', init_centroids7)


# In[23]:


#7-means with initialization
kmeans7 = KMeans(n_clusters=7, init=init_centroids7, random_state=0).fit(xnew[:,:2])
centroids7 = kmeans7.cluster_centers_
print('Final Centroids for 7-means:\n', centroids7)


# In[24]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    plt.scatter(xnew[y==val][:,0], xnew[y==val][:,1], label=val)
    plt.legend()
    plt.title('Part d\n7-means Given Initial Centroids')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.scatter(centroids7[:,0], centroids7[:,1], c='black')
plt.show()


# In[25]:


#centroids initialization for 10-means
init_centroids10 = np.zeros((10,2))
for i in range(10):
    mean10=np.expand_dims(xnew[y==i,:2].mean(0),0)
    init_centroids10[i,:] = mean10


# In[26]:


print('Initial Centroids for 10-means:\n', init_centroids10)


# In[27]:


#10-means with initialization
kmeans10 = KMeans(n_clusters=10, init=init_centroids10, random_state=0).fit(xnew[:,:2])
centroids10 = kmeans10.cluster_centers_
print('Final Centroids for 10-means:\n', centroids10)


# In[28]:


fig = plt.figure(figsize = (8,8))
for val in np.unique(y):
    plt.scatter(xnew[y==val][:,0], xnew[y==val][:,1], label=val)
    plt.legend()
    plt.title('Part d\n10-means Given Initial Centroids')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.scatter(centroids10[:,0], centroids10[:,1], c='black')
plt.show()


# ## e

# In[4]:


print('Part e')
#finding K to achieve 95 percent of the variations in PCA
s=0
k=0
while s<=0.95:
    s+=pca.explained_variance_ratio_[k]
    k+=1
print('for k= ',k ,'we have: ', s)


# In[40]:


rand = np.random.randint(x.shape[0], size=3)
fig, axs = plt.subplots(2,3,figsize=(10, 10))
r=0
for j in range(3):
    axs[0,j].set_axis_off()
    axs[0,j].imshow(x[rand[r]].reshape(28, 28))
    axs[1,j].set_axis_off()
    reconstruct = np.dot(xnew[rand[r],:k],eigvec[:k,:])
    axs[1,j].imshow(reconstruct.reshape(28, 28))
    r+=1
plt.show()


# ## f

# In[6]:


#kmeans with pca from part e
print('Part f')
kmeans = KMeans(n_clusters=10, random_state=0)
labels = kmeans.fit_predict(xnew[:,:k])


# In[13]:


fig, axs = plt.subplots(10,10,figsize=(10, 10))
for i in range(10):
    r=0
    current_cluster = x[labels==i]
    rand = np.random.randint(current_cluster.shape[0], size=10)
    for j in range(10):
        axs[i,j].set_axis_off()
        axs[i,j].imshow(current_cluster[rand[r]].reshape(28, 28))
        r+=1
plt.show()


# ## g

# In[14]:


#adding the class labels to make working in this part easier
print('Part g')
df = pd.concat([pd.DataFrame(x), pd.DataFrame(y, columns=['y'])],1)
#df.head()


# In[26]:


#plotting the bars for each cluster showing the amount of each class in the clusters
centroids = kmeans.cluster_centers_
fig, axs = plt.subplots(10,1,figsize=(5, 30))
for i in range(10):
    current_cluster = df.iloc[labels==i].reset_index(drop=True)
    percentage = list()
    for k in range(10):
        per = list(current_cluster.y).count(k)/list(y).count(k)
        percentage.append(per)
    axs[i].bar(np.arange(1,11,1),percentage)
    axs[i].set_xticks(np.arange(0,11,1))
    axs[i].set_title('Cluster %d'%i)
plt.subplots_adjust(hspace=0.4)
plt.show()


# ## h

# In[41]:


#plotting the clusters
print('Part h')
fig = plt.figure(figsize=(10, 10))
for i in range(10):
    current_cluster = xnew[labels==i,:k]
    plt.scatter(current_cluster[:,0], current_cluster[:,1], label=i)
    plt.legend()
    plt.title('Part h\n10-means')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
plt.show()


# In[43]:


fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)
for i in range(10):
    current_cluster = xnew[labels==i,:k]
    ax.scatter(current_cluster[:,0], current_cluster[:,1], current_cluster[:,2], label=i)
    ax.legend()
    ax.set_title('Part h\n10-means')
    ax.set_xlabel('First Feature')
    ax.set_ylabel('Second Feature')
    ax.set_zlabel('Third Feature')
plt.show()

