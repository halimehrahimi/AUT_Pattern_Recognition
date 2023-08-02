
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv('F://Uni/991/Pattern/SPR_HW4/inputs/P3/doughs.dat', 
                 sep="\s+", 
                 header=0, 
)
df.head()


# In[3]:


xdf = df.drop(columns='Restaurant')


# ## Computing S

# In[4]:


xdf_mean = xdf - xdf.mean()


# In[5]:


s = (df.shape[0]-1)*np.cov(xdf_mean.T)
#print(s.shape)


# ## Computing Eigenvalues and Eigenvectors

# In[6]:


eigval, eigvec = np.linalg.eig(s)
print(eigval)
print(eigvec)


# In[7]:


#This part is not necesarry, I just wrote it to show that sorting is done
eigvec = eigvec[np.argsort((-1)*eigval)]
#print(eigvec.shape)
eigval = np.sort(eigval)[::-1]
print('Eigen Values: ')
print(eigval)
print('Eigen Vectors:')
print(eigvec)


# ## a.

# In[8]:


sumlist = list()
s=0
seval = np.sum(eigval)
for k in eigval:
    sumlist.append((s+k)/seval)
print('Percentage of preservation:\n', sumlist)
print('Preserved: ',np.sum(sumlist[:3]))
print('Error: ',np.sum(sumlist[3:]))


# ## Computing Principal Components

# In[9]:


a = np.dot(xdf_mean,eigvec[:3,:].T)
#print(a.shape)
print('Principal Components:')
print(a)


# ## b.

# In[10]:


from scipy.stats import kstest


# In[11]:


print(kstest((a[:,0]-a[:,0].mean())/np.std(a[:,0]),'norm'))


# In[12]:


print(kstest((a[:,1]-a[:,1].mean())/np.std(a[:,1]),'norm'))


# In[13]:


print(kstest((a[:,2]-a[:,2].mean())/np.std(a[:,2]),'norm'))


# ## c.

# ### Mapping the Samples

# In[14]:


newx = np.dot(xdf_mean,eigvec[:3,:].T)
print(newx.shape)


# ### Plotting the samples

# In[15]:


plt.scatter(newx[:20,0], newx[:20,1])
plt.scatter(newx[20:,0], newx[20:,1])
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
plt.show()


# In[16]:


plt.scatter(newx[:20,0], newx[:20,2])
plt.scatter(newx[20:,0], newx[20:,2])
plt.xlabel('First Feature')
plt.ylabel('Third Feature')
plt.show()


# In[17]:


plt.scatter(newx[:20,1], newx[:20,2])
plt.scatter(newx[20:,1], newx[20:,2])
plt.xlabel('Second Feature')
plt.ylabel('Third Feature')
plt.show()


# In[18]:


from mpl_toolkits.mplot3d import Axes3D


# In[19]:


fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)
ax.set_xlabel('First Feature')
ax.set_ylabel('Second Feature')
ax.set_zlabel('Third Feature')
ax.scatter(newx[:20,0], newx[:20,1], newx[:20,2])
ax.scatter(newx[20:,0], newx[20:,1], newx[20:,2])

