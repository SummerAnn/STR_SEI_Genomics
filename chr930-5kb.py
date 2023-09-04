#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numpy as np
import pandas as pd


# # chr930_5kb4colnoN_3 (3window 50winnum generation)

# In[2]:


import pandas as pd
import numpy as np


df = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_subtyping/Sei/chr930/chromatin-profiles-hdf5/chr94colnoN_30_row_labels.txt", sep="\t",low_memory=False)
df


dfSei = np.load("/data/projects/nanopore/RepeatExpansion/TR_subtyping/TR_downstreamAnalysis/chr930/chr94colnoN_30.ref.raw_sequence_class_scores.npy")
dfSei = pd.DataFrame(dfSei)
dfSei



# In[3]:


df


# In[4]:


dfSei


# In[5]:


# concat axis default =0
dfinput = pd.concat([df,dfSei], axis = 1)
display(dfinput)


# In[6]:



dfinput.columns = [ "chromosome","1","window","basepair","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68"]
dfinput


display(dfinput)
dfinput = dfinput.join(dfinput['window'].str.split('_', expand=True).rename (columns={0:'TR_id', 1:'Win_num'}))
dfinput['Win_num']=dfinput['Win_num'].astype(int)
dfinput['TR_id']=dfinput['TR_id'].astype(int)
subset= dfinput[dfinput.Win_num<=5] 
subset= subset.sort_values(by=['TR_id','Win_num'])
subset

dfUpstreamdropwinnum = subset.drop_duplicates(subset=["TR_id","Win_num"], keep="first") 
print(dfUpstreamdropwinnum)

dfDownstreamdropwinnum = subset.drop_duplicates(subset=["TR_id","Win_num"], keep="last") 
print(dfDownstreamdropwinnum)


# In[7]:


dfUpstreamdropwinnum


# In[8]:


dfDownstreamdropwinnum


# In[9]:


# downstream 
dfDownstreamdropwinnum=(dfDownstreamdropwinnum.reset_index(drop=True))
dfDownwinnum = dfDownstreamdropwinnum.drop(columns=['chromosome','1','window','basepair','4','5','6','7','6','7'])
print(dfDownwinnum)
# upstream
dfUpstreamdropwinnum=(dfUpstreamdropwinnum.reset_index(drop=True))
dfUpwinnum = dfUpstreamdropwinnum.drop(columns=['chromosome','1','window','basepair','4','5','6','7','6','7'])
print(dfUpwinnum)


# # Matrix Flattening for both up and downstream

# // adding the column 'result' to iterate through. 
# // running the matrix flattening command 

# In[10]:



result = []
i = 0
for j in range(len(dfDownwinnum["TR_id"])):
   
    
    if j == len(dfDownwinnum["TR_id"])-1:
        result.append(i)
        
    elif dfDownwinnum["TR_id"].iloc[j-1] != dfDownwinnum["TR_id"].iloc[j]:
        result.append(i+1)
        i=i+1
          # if j ==0 append (i) 
    else:
        result.append(i)

dfDownwinnum["Result"] = result  
print(dfDownwinnum)


result = []
i = 0
for j in range(len(dfUpwinnum["TR_id"])):
   
    
    if j == len(dfUpwinnum["TR_id"])-1:
        result.append(i)
        
    elif dfUpwinnum["TR_id"].iloc[j-1] != dfUpwinnum["TR_id"].iloc[j]:
        result.append(i+1)
        i=i+1
          # if j ==0 append (i) 
    else:
        result.append(i)

dfUpwinnum["Result"] = result  
print(dfUpwinnum)


# In[11]:


print(dfUpwinnum)


# In[12]:


print(dfDownwinnum)
dfDownwinnum.to_csv("dfDownwinnum_chr930_5kb",index= None)
dfUpwinnum.to_csv("dfUpwinnum_chr930_5kb",index= None)


# In[13]:


#downstream
def condense_df(df, tr_id):
    #UpstreamMat = df.loc[df["TR_id"] == str(tr_id)].copy()
    DownstreamMatwinum = df.loc[df["Result"] == (tr_id)].copy()
    DownstreamMatwinum.drop("Result", axis=1,inplace=True)
    DownstreamMatwinum.drop("TR_id", axis=1,inplace=True)
    DownstreamMatwinum.drop("Win_num", axis=1,inplace=True)
    arrDown = DownstreamMatwinum.to_numpy().flatten(order='F')
    return arrDown


colsDOWN = list()
colsDOWN = dfDownwinnum.columns.tolist()
#cols_newUP = [colsUP[-1]]
print(type(colsDOWN))
print(len(colsDOWN))
#cols_newUP.extend(colsUP[0:]) 
DownstreamMatrix=dfDownwinnum
#print(UpstreamMatrix.columns)

DownstreamMatwinum = np.zeros(shape=(len(set(DownstreamMatrix["Result"].tolist())), len(condense_df(DownstreamMatrix, 1))))

print(DownstreamMatwinum.shape)
cnt = 0
failed_ids= []

for i in list(set(DownstreamMatrix["Result"].tolist())):
    cnt +=1
    try:
        DownstreamMatwinum [int (i)-1,:] = condense_df(DownstreamMatrix, i)
    except:
        failed_ids.append(i) 



DownstreamMatwinum_copy = DownstreamMatwinum
failed_ids = [i-1 for i in list(map(int,failed_ids))]
DownstreamMatdropwinum = np.delete (DownstreamMatwinum_copy, failed_ids, axis = 0)
np.save("DownstreamMatwinum_chr930_5kb",DownstreamMatdropwinum)
print(DownstreamMatdropwinum)


# In[14]:



#upstream
def condense_df(df, tr_id):
    #UpstreamMat = df.loc[df["TR_id"] == str(tr_id)].copy()
    UpstreamMatwinum = df.loc[df["Result"] == (tr_id)].copy()
    UpstreamMatwinum.drop("Result", axis=1,inplace=True)
    UpstreamMatwinum.drop("TR_id", axis=1,inplace=True)
    UpstreamMatwinum.drop("Win_num", axis=1,inplace=True)
    arrUp = UpstreamMatwinum.to_numpy().flatten(order='F')
    return arrUp


colsUp = list()
colsUp = dfUpwinnum.columns.tolist()
#cols_newUP = [colsUP[-1]]
print(type(colsUp))
print(len(colsUp))
#cols_newUP.extend(colsUP[0:]) 
UpstreamMatrix=dfUpwinnum
#print(UpstreamMatrix.columns)

UpstreamMatwinum = np.zeros(shape=(len(set(UpstreamMatrix["Result"].tolist())), len(condense_df(UpstreamMatrix, 1))))

print(UpstreamMatwinum.shape)
cnt = 0
failed_ids= []

for i in list(set(UpstreamMatrix["Result"].tolist())):
    cnt +=1
    try:
        UpstreamMatwinum [int (i)-1,:] = condense_df(UpstreamMatrix, i)
    except:
        failed_ids.append(i) 



UpstreamMatwinum_copy = UpstreamMatwinum
failed_ids = [i-1 for i in list(map(int,failed_ids))]
UpstreamMatdropwinum = np.delete (UpstreamMatwinum_copy, failed_ids, axis = 0)
np.save("UpstreamMatwinum_chr930_5kb",UpstreamMatdropwinum)
print(UpstreamMatdropwinum)


# In[15]:


# taking the log on both upstream and downstream 

log10downstreamdropwin = np.log10(DownstreamMatdropwinum)
print(log10downstreamdropwin)
log10upstreamdropwin = np.log10(UpstreamMatdropwinum)
print(log10upstreamdropwin)


# # PCA

# // downstream pca

# In[16]:


from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Downstreamdropwin= pd.DataFrame(log10downstreamdropwin)
print(Downstreamdropwin)

Downstreamdropwin.to_csv("Downstreamdropwin_chr930_5kb", index= None)
Downstreamdropwin = pd.read_csv("Downstreamdropwin_chr930_5kb")

Downstreamdropwin
x = Downstreamdropwin.values
y = Downstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfdownwinnum = PCA(n_components=8)
pcadfdownwinnummatrix = pcadfdownwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfdownwinnum.explained_variance_ratio_)
plt.savefig('pcadownstreambar_chr930_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfdownwinnum.explained_variance_ratio_)


principaldownstreamwinnum = pd.DataFrame (data = pcadfdownwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principaldownstreamwinnum['a'], principaldownstreamwinnum['b'], c='green')

plt.savefig('pcadownstream_chr930_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()


# // upstream pca

# In[17]:




Upstreamdropwin= pd.DataFrame(log10upstreamdropwin)
print(Upstreamdropwin)

Upstreamdropwin.to_csv("Upstreamdropwin_chr930_5kb.csv", index= None)
Upstreamdropwin = pd.read_csv("Upstreamdropwin_chr930_5kb.csv")

Upstreamdropwin
x = Upstreamdropwin.values
y = Upstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfupwinnum = PCA(n_components=8)
pcadfupwinnummatrix = pcadfupwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfupwinnum.explained_variance_ratio_)
plt.savefig('pcaupstreambar_chr930_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfupwinnum.explained_variance_ratio_)

principalupstreamwinnum = pd.DataFrame (data = pcadfupwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalupstreamwinnum['a'], principalupstreamwinnum['b'], c='green')
plt.savefig('pcaupstream_chr930_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()


# // save the data 

# In[18]:


principalupstreamwinnum.to_csv("principalupstreamwinnum_chr930_5kb", index= None)
principaldownstreamwinnum.to_csv("principaldownstreamwinnum_chr930_5kb", index= None)


# # TSNE

# // downstream

# In[19]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas

import sklearn

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,
              random_state=12)


n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=1000,
              random_state=12)
Z= Downstreamdropwin.values
X_2d1 = tsne.fit_transform(Z)



X_2d1 = pd.DataFrame (data = X_2d1, columns = ['a', 'b'])
plt.scatter(X_2d1 ['a'], X_2d1 ['b'], c='green')
plt.title('With perplexity = 1000, tsne for downstream Matrix after dropping winnum')
plt.savefig('tsnedownstream,perplexity:1000_chr930_5kb.pdf', dpi=299)

plt.show()


n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=50,
              random_state=12)
Z= Downstreamdropwin.values
X_2d2 = tsne.fit_transform(Z)



X_2d2 = pd.DataFrame (data = X_2d2, columns = ['a', 'b'])
plt.scatter(X_2d2 ['a'], X_2d2 ['b'], c='green')
plt.title('With perplexity = 50, tsne for downstream Matrix after dropping winnum')
plt.savefig('tsnedownstream,perplexity:50_chr930_5kb.pdf', dpi=299)
plt.show()




n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=5,
              random_state=12)
Z= Downstreamdropwin.values
X_2d = tsne.fit_transform(Z)



X_2d = pd.DataFrame (data = X_2d, columns = ['a', 'b'])
plt.scatter(X_2d ['a'], X_2d ['b'], c='green')
plt.title('With perplexity = 5, tsne for downstream Matrix after dropping winnum')
plt.savefig('tsnedownstream,perplexity:5_chr930_5kb.pdf', dpi=299)
plt.show()


# // upstream

# // save the outcomes

# In[20]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas


import sklearn

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,
              random_state=12)


n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=1000,
              random_state=12)
Z= Upstreamdropwin.values
X_2u = tsne.fit_transform(Z)



X_2u = pd.DataFrame (data = X_2u, columns = ['a', 'b'])
plt.scatter(X_2u ['a'], X_2u ['b'], c='green')
plt.title('With perplexity = 1000, tsne for upstream Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:1000_chr930_5kb.pdf', dpi=299)
plt.show()


n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=50,
              random_state=12)
Z= Upstreamdropwin.values
X_2u1 = tsne.fit_transform(Z)



X_2u1 = pd.DataFrame (data = X_2u1, columns = ['a', 'b'])
plt.scatter(X_2u1 ['a'], X_2u1 ['b'], c='green')

plt.title('With perplexity = 50, tsne for Upstreamm Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:50_chr930_5kb.pdf', dpi=299)
plt.show()

n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=5,
              random_state=12)
Z= Upstreamdropwin.values
X_2u2 = tsne.fit_transform(Z)



X_2u2 = pd.DataFrame (data = X_2u2, columns = ['a', 'b'])
plt.scatter(X_2u2 ['a'], X_2u2 ['b'], c='green')
plt.title('With perplexity = 5, tsne for Upstream Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:5_chr930_5kb.pdf', dpi=299)
plt.show()


# In[21]:


X_2u2.to_csv("X_2u2_chr930_5kb.csv", index= None)
X_2u1.to_csv("X_2u1_chr930_5kb.csv", index= None)
X_2u.to_csv("X_2u_chr930_5kb.csv", index= None)
X_2d.to_csv("X_2d_chr930_5kb.csv", index= None)
X_2d1.to_csv("X_2d1_chr930_5kb.csv", index= None)
X_2d2.to_csv("X_2d2_chr930_5kb.csv", index= None)


# # can start here after loading all the dfs required 

# In[22]:


print (dfUpwinnum)


# # SUM_TR and SEQ class Dictionary

# // upstream

# In[23]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfUpwinnum = df.loc[df["Result"] == (tr_id)].copy()
    sumTRUP =dfUpwinnum[str(9+seqclass)].sum()
    
    return sumTRUP

colsUP = list()
colsUP = dfUpwinnum.columns.tolist()
print(type(colsUP))
print(len(colsUP))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRUP= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfUpwinnum["Result"].tolist())):
    for key in translater.keys():
        sumTRUP[key].append(sum_df(dfUpwinnum,i, translater[key]))

dfUpwinnumdrop = dfUpwinnum


# // downstream

# In[24]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfDownwinnum = df.loc[df["Result"] == (tr_id)].copy()
    sumTRDOWN =dfDownwinnum[str(9+seqclass)].sum()
    
    return sumTRDOWN



colsDown = list()
colsDown = dfDownwinnum.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRDOWN= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfDownwinnum["Result"].tolist())):
    for key in translater.keys():
        sumTRDOWN[key].append(sum_df(dfDownwinnum,i, translater[key]))

dfDownwinnumdrop = dfDownwinnum


# // plotting with the seq class on pca and tsne data

# In[25]:


from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# // upstream

# // downstream

# In[26]:


# histogram
for key in translater.keys():
    plt.hist(sumTRDOWN[key])
    plt.title("Bar, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream_chr930_5kb.pdf",dpi=299)
    plt.show()
    
for key in translater.keys():
    plt.plot(sumTRDOWN[key])
    plt.title(key)
    plt.show()
    
# chr930_5kb_down_tsne, perplex: 5
for key in translater.keys():

    X_2d[key] = sumTRDOWN[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream_tsne:5_chr930_5kb.pdf",dpi=299)
    plt.show()
    
# chr930_5kb_down_tsne, perplex: 50
for key in translater.keys():

    X_2d1[key] = sumTRDOWN[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d1, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream_tsne:50_chr930_5kb.pdf",dpi=299)
    plt.show()
 
 # chr930_5kb_down_pca  
for key in translater.keys():

    principaldownstreamwinnum[key] = sumTRDOWN[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue=key)
    plt.title("pca, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream_pca_chr930_5kb.pdf",dpi=299)
    plt.show()


# # Most Contributing 3 and 10 

# // 3  upstream

# In[27]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

most_important3_differentmethod=np.abs(pca.components_).argsort()[::-1][:3]

mostimportant3= most_important3_differentmethod[:,0]
mostimportant3= list(mostimportant3)

#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

most_important3_differentmethod=np.abs(pca.components_).argsort()[::-1][:3]

mostimportant3= most_important3_differentmethod[:,0]
mostimportant3= list(mostimportant3)

Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns 
mostimportantsubset=Upstreamdropwin[mostimportant3]
mostimportantsubset

from sklearn.decomposition import PCA
pcadfupstreamMatreal = PCA(n_components=2)
principalComponentsdfupstreamMatreal = pcadfupstreamMatreal.fit_transform(Upstreamdropwin)

plt.bar(x=range(2), height= pcadfupstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfupstreamMatreal.explained_variance_ratio_)

principalupstreamDfreal = pd.DataFrame (data = principalComponentsdfupstreamMatreal, columns = ['a', 'b'])
plt.scatter(principalupstreamDfreal['a'], principalupstreamDfreal['b'], c='purple')
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principalupstreamwinnum[['mostimportant_1','most_important_2','mostimportant_3']] = mostimportantsubset

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue="mostimportant_1")


# // 3 downstream

# In[28]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Downstreamdropwin)

most_important3ones_differentmethod=np.abs(pca.components_).argsort()[::-1][:3]

mostimportant3ones= most_important3ones_differentmethod[:,0]
mostimportant3ones= list(mostimportant3)
Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 

mostimportantsubsetDown=Downstreamdropwin[mostimportant3ones]
mostimportantsubsetDown

from sklearn.decomposition import PCA
pcadfdownstreamMatreal = PCA(n_components=2)
principalComponentsdfdownstreamMatreal = pcadfdownstreamMatreal.fit_transform(Downstreamdropwin)

plt.bar(x=range(2), height= pcadfdownstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfdownstreamMatreal.explained_variance_ratio_)

principaldownstreamDfreal = pd.DataFrame (data = principalComponentsdfdownstreamMatreal, columns = ['a', 'b'])
plt.scatter(principaldownstreamDfreal['a'], principaldownstreamDfreal['b'], c='green')
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principaldownstreamwinnum[['mostimportant_1','most_important_2','mostimportant_3']] = mostimportantsubsetDown

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue="mostimportant_1")


# # UMAP

# In[29]:


import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('conda install -c conda-forge umap-learn -y')

# Data manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Visualization
import plotly.express as px # for data visualization
import matplotlib.pyplot as plt # for showing handwritten digits

# Skleran
from sklearn.datasets import load_digits # for MNIST data
from sklearn.model_selection import train_test_split # for splitting data into train and test samples

# UMAP dimensionality reduction
from umap import UMAP
import umap
get_ipython().system('pip install umap-learn')


# // save the files

# In[30]:


Upstreamdropwin.to_csv("Upstreamdropwin_chr930_5kb.tsv", index= None)
Downstreamdropwin.to_csv("Downstreamdropwin_chr930_5kb.tsv", index= None)


# In[31]:


Upstreamdropwin.to_csv("Upstreamdropwin_chr930_5kb.tsv", index= None)
Downstreamdropwin.to_csv("Downstreamdropwin_chr930_5kb.tsv", index= None)
Upstreamdropwin = pd.read_csv("Upstreamdropwin_chr930_5kb.tsv")
Downstreamdropwin= pd.read_csv("Downstreamdropwin_chr930_5kb.tsv")


# // downstream

# # downstream umap full

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# downstream fullmatrix
Downstreamdropwin
x = Downstreamdropwin.values
y = Downstreamdropwin.values
x = StandardScaler().fit_transform(x)

downstreamumap = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=2, # default 2, The dimension of the space to embed into.
               metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
               n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
               learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
               init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
               min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
               spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
               low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
               set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
               local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
               repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
               negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
               transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
               a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
               metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
               angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
               target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
               #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different. 
               #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
               #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
               transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
               verbose=False, # default False, Controls verbosity of logging.
               unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded. 
              )

# Fit and transform the data
Xdown = downstreamumap.fit_transform(x)

# Check the shape of the new data
print('Shape of X_trans: ', Xdown.shape)
Xdowndf=pd.DataFrame(Xdown)

plt.scatter(x=Xdowndf[0],y=Xdowndf[1])
plt.show()


## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=Xdowndf,x=Xdowndf[0],y=Xdowndf[1], hue=Xdowndf[0])

for key in translater.keys():

    Xdowndf[key] = sumTRDOWN[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdowndf, x=Xdowndf[0],y=Xdowndf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream data-prereduction_chr930_5kb")
    plt.savefig("seqClass" + key+ "DOwnstream_umap(prereduct)_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[33]:


plt.hist(Xdowndf.astype(float))
umap2dimensiondown_chr930_5kb=pd.DataFrame(Xdown)
print(umap2dimensiondown_chr930_5kb)


# # plt sum of seq TR downstream 

# In[34]:


# downstream plt sum of seq TR downstream 

def sum_df(df, tr_id, seqclass):
    #print (df)
    dfDownwinnum = df.loc[df["Result"] == (tr_id)].copy()
    sumTRpltdown =dfDownwinnum[str(seqclass)].sum()
    return sumTRpltdown



colsDown = list()
colsDown = dfDownwinnum.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)
sumTRpltdown= {}

cnt = 0

for tr_id in list(set(dfDownwinnum["Result"].tolist())):
    sumTRpltdown[tr_id]=[]
    for seqclass in range(9,69):
        
        sumTRpltdown[tr_id].append(sum_df(dfDownwinnum,tr_id, seqclass))
    dfDownwinnumdrop = dfDownwinnum
    
for tr_id in range(9,69):
    plt.plot(sumTRpltdown[tr_id])
    plt.title(tr_id)
    plt.show()
    
dfsumTRpltdown=pd.DataFrame(sumTRpltdown).T


x = dfsumTRpltdown.values
y = dfsumTRpltdown.values
x = StandardScaler().fit_transform(x)

dfsumTRpltdownumap = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=2, # default 2, The dimension of the space to embed into.
               metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
               n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
               learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
               init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
               min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
               spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
               low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
               set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
               local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
               repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
               negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
               transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
               a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
               metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
               angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
               target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
               #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different. 
               #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
               #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
               transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
               verbose=False, # default False, Controls verbosity of logging.
               unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded. 
              )

# Fit and transform the data
Xdownumap = dfsumTRpltdownumap.fit_transform(x)

# Check the shape of the new data
print('Shape of X_trans: ', Xdownumap.shape)

Xdownumapdf=pd.DataFrame(Xdownumap)

for key in translater.keys():

    Xdownumapdf[key] = sumTRDOWN [key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdownumapdf, x=Xdownumapdf[0],y=Xdownumapdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream data_chr930_5kb")
    plt.savefig("seqClass" + key+ "downstream_umap_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[35]:


reductdownumap2d=pd.DataFrame(Xdownumap)


# // upstream

# # uptream umap full

# In[36]:


x = Upstreamdropwin.values
y = Upstreamdropwin.values
x = StandardScaler().fit_transform(x)

Upstreamumap = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=2, # default 2, The dimension of the space to embed into.
               metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
               n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
               learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
               init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
               min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
               spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
               low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
               set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
               local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
               repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
               negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
               transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
               a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
               metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
               angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
               target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
               #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different. 
               #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
               #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
               transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
               verbose=False, # default False, Controls verbosity of logging.
               unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded. 
              )

# Fit and transform the data
XUp = Upstreamumap.fit_transform(x)

# Check the shape of the new data
print('Shape of X_trans: ', XUp.shape)

XUpdf=pd.DataFrame(XUp)

for key in translater.keys():

    XUpdf[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=XUpdf, x=XUpdf[0],y=XUpdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Upstream dataaaa-prereduction_chr930_5kb")
    plt.savefig("seqClass" + key+ "Upstream_umap(prereduct)_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[37]:


plt.hist(XUpdf.astype(float))
umap2dimensionup_chr930_5kb=pd.DataFrame(XUp)
print(umap2dimensionup_chr930_5kb)


# # plt sum of seq TR upstream 

# In[38]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfUpwinnum = df.loc[df["Result"] == (tr_id)].copy()
    sumTRpltup =dfUpwinnum[str(seqclass)].sum()
    return sumTRpltup


colsUp = list()
colsUp = dfUpwinnum.columns.tolist()
print(type(colsUp))
print(len(colsUp))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)
sumTRpltup = {}

cnt = 0

for tr_id in list(set(dfUpwinnum["Result"].tolist())):
    sumTRpltup[tr_id]=[]
    for seqclass in range(9,69):
        
        sumTRpltup[tr_id].append(sum_df(dfUpwinnum,tr_id, seqclass))
    dfUpwinnumdrop = dfUpwinnum
    
dfsumTRpltup=pd.DataFrame(sumTRpltup).T


x = dfsumTRpltup.values
y = dfsumTRpltup.values
x = StandardScaler().fit_transform(x)

dfsumTRpltupumap = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=2, # default 2, The dimension of the space to embed into.
               metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
               n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
               learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
               init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
               min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
               spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
               low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
               set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
               local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
               repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
               negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
               transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
               a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
               metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
               angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
               target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
               #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different. 
               #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
               #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
               transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
               verbose=False, # default False, Controls verbosity of logging.
               unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded. 
              )

# Fit and transform the data
Xupumap = dfsumTRpltupumap.fit_transform(x)

# Check the shape of the new data
print('Shape of X_trans: ', Xupumap.shape)

Xupumapdf=pd.DataFrame(Xupumap)

for key in translater.keys():

    Xupumapdf[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xupumapdf, x=Xupumapdf[0],y=Xupumapdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Upstream data_chr930_5kb")
    plt.savefig("seqClass" + key+ "Upstream_umap_chr930_5kb.pdf",dpi=299)
    plt.show()
    
reductupumap2d_chr930_5kb=pd.DataFrame(Xupumap)


# In[39]:


# save the dfs
dfsumTRpltup.to_csv("dfsumTRpltup_chr930_5kb.tsv", index= None)
dfsumTRpltdown.to_csv("dfsumTRpltdown_chr930_5kb.tsv", index= None)
umap2dimensionup_chr930_5kb.to_csv("umap2dimensionup_chr930_5kb.tsv", index= None)
umap2dimensiondown_chr930_5kb.to_csv("umap2dimensiondown_chr930_5kb.tsv", index= None)
# reduced 
reductdownumap2d.to_csv("reductdownumap2d_chr930_5kb.tsv", index= None)
reductupumap2d_chr930_5kb.to_csv("reductupumap2d_chr930_5kb.tsv", index= None)


# # Kmean

# // upstream (fullmatrix,pca,tsne,umap)

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

Upstreamdropwinscale=StandardScaler().fit_transform(Upstreamdropwin)

#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(Upstreamdropwinscale)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

#instantiate the k-means class, using optimal number of clusters
kmeans = KMeans(init="random", n_clusters=3, n_init=10, random_state=1)

#fit k-means algorithm to data
kmeans.fit(Upstreamdropwinscale)

#view cluster assignments for each observation
kmeans.labels_

upscaledff=pd.DataFrame(Upstreamdropwinscale)

#append cluster assingments to original DataFrame
upscaledff['cluster'] = kmeans.labels_

#view updated DataFrame
print(upscaledff)

upscaledff['cluster'].unique()

for key in translater.keys():

    upscaledff[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=upscaledff, x=upscaledff[0], y=upscaledff[1], hue=key)
    plt.title(key)
    plt.show()
    
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=upscaledff, x=upscaledff[0], y=upscaledff[1], hue='cluster')
plt.title("kmeans, Upstream data")
plt.savefig("kmeans,Upstream_fullmatrix_chr930_5kb.pdf",dpi=299)
plt.show()


# In[41]:


# on PCA
principalupstreamwinnumscale=StandardScaler().fit_transform(principalupstreamwinnum)
#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(principalupstreamwinnumscale)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

principalupstreamwinnumscaledf=pd.DataFrame(principalupstreamwinnumscale)

#append cluster assingments to original DataFrame\
principalupstreamwinnumscaledf=pd.DataFrame(principalupstreamwinnumscale)
principalupstreamwinnumscaledf['cluster'] = kmeans.labels_

#view updated DataFrame
print(principalupstreamwinnumscaledf)

for key in translater.keys():

    principalupstreamwinnumscaledf[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnumscaledf, x=principalupstreamwinnumscaledf[0], y=principalupstreamwinnumscaledf[1], hue=key)
    plt.title(key)
    plt.show()
    
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnumscaledf, x=principalupstreamwinnumscaledf[0], y=principalupstreamwinnumscaledf[1], hue='cluster')
plt.title("kmeans, Upstream pca dataa_chr930_5kb")
plt.savefig("kmeans,Upstream_pca_chr930_5kb.pdf",dpi=299)
plt.show()


# In[42]:


tsneupscaledf=StandardScaler().fit_transform(X_2u)
#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(tsneupscaledf)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

tsneupscaledf
#append cluster assingments to original DataFrame\
tsneupscaledf=pd.DataFrame(tsneupscaledf)
tsneupscaledf['cluster'] = kmeans.labels_

#view updated DataFrame
print(tsneupscaledf)

tsneupscaledf['cluster'] = kmeans.labels_

tsneupscaledf

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsneupscaledf, x=tsneupscaledf[0], y=tsneupscaledf[1], hue='cluster')
plt.title("kmeans, Upstream dataaa tsne_chr930_5kb")
plt.savefig("kmeans,Upstream_tsne_chr930_5kb.pdf",dpi=299)
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsneupscaledf[tsneupscaledf['cluster']== 2], x=tsneupscaledf[tsneupscaledf['cluster']== 2][0], y=tsneupscaledf[tsneupscaledf['cluster']== 2][1], hue='cluster')
plt.title("tsne, kmean")
plt.show()


# In[43]:


# on UMAP
dfsumTRpltupscale=StandardScaler().fit_transform(umap2dimensionup_chr930_5kb)
#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(dfsumTRpltupscale)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

dfsumTRpltupscalef=pd.DataFrame(dfsumTRpltupscale)



dfsumTRpltupscalef['cluster'] = kmeans.labels_

#view updated DataFrame
print(dfsumTRpltupscalef)

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=dfsumTRpltupscalef, x=dfsumTRpltupscalef[0], y=dfsumTRpltupscalef[1], hue='cluster')
plt.title("kmeans, Upstream dataaa umap_chr930_5kb")
plt.savefig("kmeans,Upstream_umap_chr930_5kb.pdf",dpi=299)
plt.show()


# // downstream (fullmatrix,pca,tsne,umap)

# In[44]:


downstreamdropwinscale=StandardScaler().fit_transform(Downstreamdropwin)
#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(downstreamdropwinscale)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

#instantiate the k-means class, using optimalxdowndf.tsver of clusters
kmeans = KMeans(init="random", n_clusters=3, n_init=10, random_state=1)

#fit k-means algorithm to data
kmeans.fit(downstreamdropwinscale)

#view cluster assignments for each observation
kmeans.labels_

downscaledff=pd.DataFrame(downstreamdropwinscale)
#append cluster assingments to original DataFrame
downscaledff['cluster'] = kmeans.labels_

downscaledff.to_csv("downscaledff", index= None)
downscaledff
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=downscaledff, x=downscaledff[0], y=downscaledff[1], hue='cluster')
plt.title("kmean, fullmatrix_Downstream dataaaa_chr930_5kb")
plt.savefig("downstream_kmean_fullmatrix_chr930_5kb.pdf",dpi=299)
plt.show()


# In[45]:


# on PCA
principaldownstreamwinnumscale=StandardScaler().fit_transform(principaldownstreamwinnum)
#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(principaldownstreamwinnumscale)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

#append cluster assingments to original DataFrame\
principaldownstreamwinnumscaledf=pd.DataFrame(principaldownstreamwinnumscale)
principaldownstreamwinnumscaledf['cluster'] = kmeans.labels_

#view updated DataFrame
print(principaldownstreamwinnumscaledf)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnumscaledf, x=principaldownstreamwinnumscaledf[0], y=principaldownstreamwinnumscaledf[1], hue='cluster')
plt.title("kmean, pca_Downstream dataaa_chr930_5kb")
plt.savefig("downstream_kmean_pca_chr930_5kb.pdf",dpi=299)
plt.show()


# In[46]:


tsnedownscaledf=StandardScaler().fit_transform(X_2d)
#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(tsnedownscaledf)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

tsnedownscaledf=pd.DataFrame(tsnedownscaledf)
tsnedownscaledf['cluster'] = kmeans.labels_
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsnedownscaledf, x=tsnedownscaledf[0], y=tsnedownscaledf[1], hue='cluster')
plt.title("kmean, tsne_Downstream dataa_chr930_5kb")
plt.savefig("downstream_kmean_tsne_chr930_5kb.pdf",dpi=299)
plt.show()


# In[47]:


# on UMAP
Xdownumapdfscale=StandardScaler().fit_transform(umap2dimensiondown_chr930_5kb)
#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(Xdownumapdfscale)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

Xdownumapdfscale=pd.DataFrame(Xdownumapdfscale)
Xdownumapdfscale
Xdownumapdfscale['cluster'] = kmeans.labels_

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data= Xdownumapdfscale, x= Xdownumapdfscale[0], y= Xdownumapdfscale[1], hue='cluster')
plt.title("kmean, umap_Downstream dataaa_chr930_5kb")
plt.savefig("downstream_kmean_umap_chr930_5kb.pdf",dpi=299)
plt.show()


# In[48]:


# all the needed import untill ouvain 
import numpy as np
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('conda install -c conda-forge umap-learn -y')

# Data manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Visualization
import plotly.express as px # for data visualization
import matplotlib.pyplot as plt # for showing handwritten digits

# Skleran
from sklearn.datasets import load_digits # for MNIST data
from sklearn.model_selection import train_test_split # for splitting data into train and test samples

# UMAP dimensionality reduction
from umap import UMAP
import umap
get_ipython().system('pip install umap-learn')

from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn

from sklearn.manifold import TSNE


# In[49]:


# all the needed import untill ouvain 
import numpy as np
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('conda install -c conda-forge umap-learn -y')

# Data manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Visualization
import plotly.express as px # for data visualization
import matplotlib.pyplot as plt # for showing handwritten digits

# Skleran
from sklearn.datasets import load_digits # for MNIST data
from sklearn.model_selection import train_test_split # for splitting data into train and test samples

# UMAP dimensionality reduction
from umap import UMAP
import umap
get_ipython().system('pip install umap-learn')

from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn

from sklearn.manifold import TSNE

from umap import UMAP
import umap
get_ipython().system('pip install umap-learn')

from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn

from sklearn.manifold import TSNE

# tsne
X_2u2=  pd.read_csv("X_2u2_chr930_5kb.csv")
X_2u1= pd.read_csv("X_2u1_chr930_5kb.csv")
X_2u= pd.read_csv("X_2u_chr930_5kb.csv")
X_2d= pd.read_csv("X_2d_chr930_5kb.csv")
X_2d1= pd.read_csv("X_2d1_chr930_5kb.csv")
X_2d2= pd.read_csv("X_2d2_chr930_5kb.csv")
# pca
principalupstreamwinnum= pd.read_csv("principalupstreamwinnum_chr930_5kb")
principaldownstreamwinnum= pd.read_csv("principaldownstreamwinnum_chr930_5kb")
#full matrix
dfDownwinnum= pd.read_csv("dfDownwinnum_chr930_5kb")
dfUpwinnum= pd.read_csv("dfUpwinnum_chr930_5kb")
Upstreamdropwin = pd.read_csv("Upstreamdropwin_chr930_5kb.csv")
Downstreamdropwin = pd.read_csv("Downstreamdropwin_chr930_5kb")
#louvain
upresult= pd.read_csv("upresult_chr930_5kb.tsv")
downresult= pd.read_csv("downresult_chr930_5kb.tsv")
dfupprediction= pd.read_csv("dfupprediction_chr930_5kb.tsv")
dfdownprediction= pd.read_csv("dfdownprediction_chr930_5kb.tsv")
# save the dfs
dfsumTRpltup= pd.read_csv("dfsumTRpltup_chr930_5kb.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chr930_5kb.tsv")
umap2dimensionup_chr9= pd.read_csv("umap2dimensionup_chr930_5kb.tsv")
umap2dimensiondown_chr9= pd.read_csv("umap2dimensiondown_chr930_5kb.tsv")
# reduced 
reductdownumap2d= pd.read_csv("reductdownumap2d_chr930_5kb.tsv")
reductupumap2d_chr9= pd.read_csv("reductupumap2d_chr930_5kb.tsv")
# louvain byw
dfdownpredictionbyw= pd.read_csv("downpredictionbyw_chr930_5kb.tsv")
dfuppredictionbyw= pd.read_csv("uppredictionbyw_chr930_5kb.tsv")


# # Louvain

# In[50]:


get_ipython().system(' pip install igraph')
get_ipython().system(' pip install louvain')

import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_rand_score
import scipy
from tqdm import tqdm
from sklearn import preprocessing
import networkx as nx
import community
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import louvain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.ion()
plt.show()


# In[51]:


n_clusters = 6
n_features=305
n_samples=19939
random_state = 42


updata, truth = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state, n_features=n_features)
updata = preprocessing.MinMaxScaler().fit_transform(Upstreamdropwin)

# Plot
plt.scatter(updata[:, 0], updata[:, 1], s=50, c = truth, cmap = 'viridis')
plt.title(f"Example of a mixture of {n_clusters} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2");

n_clusters = 6
n_features=305
n_samples=19939
random_state = 42


downdata, truth = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state, n_features=n_features)
downdata = preprocessing.MinMaxScaler().fit_transform(Downstreamdropwin)
downdata 

# Plot
plt.scatter(downdata[:, 0], downdata[:, 1], s=50, c = truth, cmap = 'viridis')
plt.title(f"Example of a mixture of {n_clusters} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2");


# In[52]:


def cluster_by_connectivity(downdata, neighbors = 10, resolution_parameter = 1):
    """
    This method partitions input data by applying the louvain algorithm
    on the connectivity binary matrix returned by the kneighbors graph.
  

  """
    A = kneighbors_graph(downdata, neighbors, mode='connectivity', include_self=True)
    sources, targets = A.nonzero()
    weights = A[sources, targets]
    if isinstance(weights, np.matrix): # ravel data
        weights = weights.A1
    g = ig.Graph(directed=False)
    g.add_vertices(A.shape[0])  # each observation is a node
    edges = list(zip(sources, targets))
    g.add_edges(edges)
    
    g.es['weight'] = weights
    weights = np.array(g.es["weight"]).astype(np.float64)
    partition_type = louvain.RBConfigurationVertexPartition
    partition_kwargs = {}
    partition_kwargs["weights"] = weights
    partition_kwargs["resolution_parameter"] = resolution_parameter
    part = louvain.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)
    return groups

def cluster_by_connectivity(updata, neighbors = 10, resolution_parameter = 1):
    """
    This method partitions input data by applying the louvain algorithm
    on the connectivity binary matrix returned by the kneighbors graph.
  

  """
    A = kneighbors_graph(updata, neighbors, mode='connectivity', include_self=True)
    sources, targets = A.nonzero()
    weights = A[sources, targets]
    if isinstance(weights, np.matrix): # ravel data
        weights = weights.A1
    g = ig.Graph(directed=False)
    g.add_vertices(A.shape[0])  # each observation is a node
    edges = list(zip(sources, targets))
    g.add_edges(edges)
    
    g.es['weight'] = weights
    weights = np.array(g.es["weight"]).astype(np.float64)
    partition_type = louvain.RBConfigurationVertexPartition
    partition_kwargs = {}
    partition_kwargs["weights"] = weights
    partition_kwargs["resolution_parameter"] = resolution_parameter
    part = louvain.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)
    return groups

downpredictionbyw = cluster_by_connectivity(downdata, resolution_parameter = 1)
Counter(downpredictionbyw)

uppredictionbyw = cluster_by_connectivity(updata, resolution_parameter = 1)
Counter(uppredictionbyw)


# In[53]:



cnt = 0
resolution_parameter = {'0.3': 0.3, '0.5': 0.5, '0.6': 0.6, '0.7': 0.7, '0.8': 0.8, '0.9': 0.9,'1': 1}
resolution_result ={}
for key in resolution_parameter.keys():
    resolution_result[key]  = cluster_by_connectivity(downdata, resolution_parameter = resolution_parameter[key])
    

cnt = 0
resolution_parameter = {'0.3': 0.3, '0.5': 0.5, '0.6': 0.6, '0.7': 0.7, '0.8': 0.8, '0.9': 0.9,'1': 1}
resolution_result ={}
for key in resolution_parameter.keys():
    resolution_result[key]  = cluster_by_connectivity(updata, resolution_parameter = resolution_parameter[key])


# In[54]:


downdistanceMatrix =  euclidean_distances(downdata, downdata)
print(downdistanceMatrix.shape)
updistanceMatrix =  euclidean_distances(updata, updata)
print(updistanceMatrix.shape)


# In[55]:


def cluster_by_distance_matrix(updistanceMatrix, resolution_parameter = 1.5):
    """
    This method partitions input data by applying the louvain algorithm
    on a given distance matrix.
    A similarity matrix is computed from the distance matrix and its elements
    will serve as edge weights.
    """
    # convert distance matrix to similariy matrix
    updistanceMatrix = 1- updistanceMatrix/np.max(updistanceMatrix)
    edges = np.unravel_index(np.arange(updistanceMatrix.shape[0]*updistanceMatrix.shape[1]), updistanceMatrix.shape)
    edges = list(zip(*edges))
    weights = updistanceMatrix.ravel()
    
    g = ig.Graph(directed=False)
    g.add_vertices(updistanceMatrix.shape[0])  # each observation is a node
    g.add_edges(edges)
    
    g.es['weight'] = weights
    weights = np.array(g.es["weight"]).astype(np.float64)
    partition_type = louvain.RBConfigurationVertexPartition
    partition_kwargs = {}
    partition_kwargs["weights"] = weights
    partition_kwargs["resolution_parameter"] = resolution_parameter
    part = louvain.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)
    return groups


# In[56]:


def cluster_by_distance_matrix(downdistanceMatrix, resolution_parameter = 1.5):
    """
    This method partitions input data by applying the louvain algorithm
    on a given distance matrix.
    A similarity matrix is computed from the distance matrix and its elements
    will serve as edge weights.
    """
    # convert distance matrix to similariy matrix
    downdistanceMatrix = 1- downdistanceMatrix/np.max(downdistanceMatrix)
    edges = np.unravel_index(np.arange(downdistanceMatrix.shape[0]*downdistanceMatrix.shape[1]), downdistanceMatrix.shape)
    edges = list(zip(*edges))
    weights = downdistanceMatrix.ravel()
    
    g = ig.Graph(directed=False)
    g.add_vertices(downdistanceMatrix.shape[0])  # each observation is a node
    g.add_edges(edges)
    
    g.es['weight'] = weights
    weights = np.array(g.es["weight"]).astype(np.float64)
    partition_type = louvain.RBConfigurationVertexPartition
    partition_kwargs = {}
    partition_kwargs["weights"] = weights
    partition_kwargs["resolution_parameter"] = resolution_parameter
    part = louvain.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)
    return groups


# In[57]:


downprediction = cluster_by_distance_matrix(downdistanceMatrix)
Counter(downprediction)


# In[58]:


upprediction = cluster_by_distance_matrix(updistanceMatrix)
Counter(upprediction)



# In[59]:


adjusted_rand_score(truth, downprediction)
adjusted_rand_score(truth, upprediction)


# In[60]:


dfdownprediction=pd.DataFrame(downprediction)
print(dfdownprediction)

dfupprediction=pd.DataFrame(upprediction)
print(dfupprediction)


# In[61]:



downresult = pd.concat([Downstreamdropwin, dfdownprediction], axis=1)
downresult.columns = [*downresult.columns[:-1], 'down_prediction']
upresult = pd.concat([Upstreamdropwin, dfupprediction], axis=1)
upresult.columns = [*upresult.columns[:-1], 'up_prediction']


# In[62]:



upresult.to_csv("upresult_chr930_5kb.tsv", index= None)
downresult.to_csv("downresult_chr930_5kb.tsv", index= None)
dfupprediction.to_csv("dfupprediction_chr930_5kb.tsv", index= None)
dfdownprediction.to_csv("dfdownprediction_chr930_5kb.tsv", index= None)
dfdownpredictionbyw=pd.DataFrame(downpredictionbyw)
dfuppredictionbyw=pd.DataFrame(uppredictionbyw)
dfdownpredictionbyw.to_csv("downpredictionbyw_chr930_5kb.tsv", index= None)
dfuppredictionbyw.to_csv("uppredictionbyw_chr930_5kb.tsv", index= None)


# In[63]:


upresult = pd.read_csv("upresult_chr930_5kb.tsv")
downresult= pd.read_csv("downresult_chr930_5kb.tsv")


# In[64]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=upresult, x=upresult['0'], y=upresult['1'], hue='up_prediction')
plt.title("up louvain")
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=downresult, x=downresult['0'], y=downresult['1'], hue='down_prediction')
plt.title("down louvain")
plt.show()


# In[65]:


for key in resolution_parameter.keys():

    principalupstreamwinnum[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue=key, palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream data")
    plt.savefig("louvain_res_" + key+ "_pca_up_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[66]:


for key in resolution_parameter.keys():

    principaldownstreamwinnum[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue=key, palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Downstream data")
    plt.savefig("louvain_res_" + key+ "_pca_down_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[67]:



dfsumTRpltupscalef=pd.DataFrame(dfsumTRpltupscale)
dfsumTRpltupscalef['cluster'] = kmeans.labels_

#view updated DataFrame
print(dfsumTRpltupscalef)
for key in resolution_parameter.keys():

    dfsumTRpltupscalef[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=dfsumTRpltupscalef, x=dfsumTRpltupscalef[0],y=dfsumTRpltupscalef[1], hue= resolution_result[key])
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream dataaa")
    plt.savefig("louvain_res_" + key+ "_umap_up_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[68]:



Xdownumapdf=pd.DataFrame(dfsumTRpltupscale)
Xdownumapdf['cluster'] = kmeans.labels_

#view updated DataFrame
print(Xdownumapdf)
for key in resolution_parameter.keys():
    Xdownumapdf[key] = resolution_result[key]
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data= Xdownumapdf, x= Xdownumapdf[0],y= Xdownumapdf[1], hue= resolution_result[key])
plt.title("Louvain Clustering, resolution: "+ key + ", Upstream dataaa")
plt.savefig("louvain_res_" + key+ "_umap_up_chr930_5kb.pdf",dpi=299)
plt.show()


# In[69]:


for key in resolution_parameter.keys():

    reductupumap2d_chr9[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue=key, palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream data")
    plt.savefig("louvain_res_" + key+ "_umap_sumTR_up_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[70]:


for key in resolution_parameter.keys():

    reductdownumap2d[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue=key,palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Downstream data")
    plt.savefig("louvain_res_" + key+ "_umap_sumTR_down_chr930_5kb.pdf",dpi=299)
    plt.show()


# # chr9ch38 validation coloring

# // revise this
# import numpy as np
# import pandas as pd
# ch38 = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/closestedsortedchr930_5kbch38.bed", sep="\t", header=None, names = ['chr9','1','2','chr91','3','4','class','distance'])
# ch38

# In[71]:


import numpy as np
import pandas as pd
ch38 = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/closestedsortedchr9ch38.bed", sep="\t", header=None, names = ['chr9','1','2','chrX9','3','4','class','distance'])
ch38
ch38drop = ch38.drop_duplicates(subset=["1","2"],keep="first")


# # filter by the reduct dimentionality technique
# 

# In[72]:


# pca downstream 
principaldownstreamwinnum['class'] =ch38drop['class']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='class', palette="tab10")
plt.title("downstream pca with ch38")
plt.show()

# pca upstream
principalupstreamwinnum['class'] =ch38drop['class']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='class', palette="tab10")
plt.title("upstream pca with ch38")
plt.show()


# # filter by distance 

# In[73]:


# filter by distance downstream


# In[74]:


# 2000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<2000)&(ch38drop['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 2000')
plt.savefig("Downstreampcawithvalidationdatadistance<000_ch9.pdf",dpi=299)
plt.show()



# 50000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<50000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 50000')
plt.savefig("Downstreampcawithvalidationdatadistance<50000_ch9.pdf",dpi=299)
plt.show()


# 5000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<5000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 5000')
plt.savefig("Downstreampcawithvalidationdatadistance<5000_chr930_5kb.pdf",dpi=299)
plt.show()


# 1000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<1000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 1000')
plt.savefig("Downstreampcawithvalidationdatadistance<1000_chr930_5kb.pdf",dpi=299)
plt.show()

# 500
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<500)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 500')
plt.savefig("Downstreampcawithvalidationdatadistance<500_chr930_5kb.pdf",dpi=299)
plt.show()


# In[75]:


# filter by distance upstream


# In[76]:


X_2u['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='distance', palette="tab10")
plt.title( 'Upstream TSNE with ch38drop distance of larger than -2000')
plt.savefig("Upstream TSNE withvalidationdatadistance>-2000_chr930_5kb.pdf",dpi=299)
plt.show()

X_2u1['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u1, x="a", y="b", hue='distance', palette="tab10")
plt.title('Upstream TSNE:50 with ch38drop distance of larger than -2000')
plt.savefig("Upstream TSNE:50 withvalidationdatadistance>-2000_chr930_5kb.pdf",dpi=299)
plt.show()

X_2u['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='distance', palette="tab10")
plt.title('Upstream TSNE with ch38drop distance of larger than -2000')
plt.savefig("Upstream TSNE withvalidationdatadistance>-2000_chr930_5kb.pdf",dpi=299)
plt.show()

principalupstreamwinnum['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Upstream pca with ch38drop distance of larger than -2000')
plt.savefig("Upstreampcawithvalidationdatadistance>-2000_chr930_5kb.pdf",dpi=299)
plt.show() 

principalupstreamwinnum['distance'] = np.where((ch38drop['distance']>-1000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Upstream pca with ch38drop distance of larger than -1000')
plt.savefig("Upstreampcawithvalidationdatadistance>-1000_chr930_5kb.pdf",dpi=299)
plt.show()


# # CTCF coloring 

# In[77]:


ch38drop['CTCFbound']= None
ch38drop

#Upsteam tsne
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('DNase-H3K4me3,CTCF-bound'),'DNase-H3K4me3', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('PLS,CTCF-bound'),'PLS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('pELS,CTCF-bound'),'pELS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('dELS,CTCF-bound'),'dELS', ch38drop['CTCFbound'])
X_2u['CTCFbound'] = ch38drop['CTCFbound']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='CTCFbound', palette="tab10")
plt.title('Upstream TSNE with ch38drop class CTCFbound')
plt.show()


# In[78]:


#Downsteam tsne
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('DNase-H3K4me3,CTCF-bound'),'DNase-H3K4me3', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('PLS,CTCF-bound'),'PLS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('pELS,CTCF-bound'),'pELS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('dELS,CTCF-bound'),'dELS', ch38drop['CTCFbound'])
X_2d['CTCFbound'] = ch38drop['CTCFbound']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='CTCFbound', palette="tab10")
plt.title('Downstream TSNE with ch38drop class CTCFbound')
plt.show()


# In[79]:


#downsteam pca
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('DNase-H3K4me3,CTCF-bound'),'DNase-H3K4me3', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('PLS,CTCF-bound'),'PLS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('pELS,CTCF-bound'),'pELS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('dELS,CTCF-bound'),'dELS', ch38drop['CTCFbound'])
principaldownstreamwinnum['CTCFbound'] = ch38drop['CTCFbound']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='CTCFbound', palette="tab10")
plt.title('Downstream principaldownstreamwinnum with ch38drop class CTCFbound')
plt.show()


# In[80]:


#upsteam pca
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('DNase-H3K4me3,CTCF-bound'),'DNase-H3K4me3', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('PLS,CTCF-bound'),'PLS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('pELS,CTCF-bound'),'pELS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('dELS,CTCF-bound'),'dELS', ch38drop['CTCFbound'])
principalupstreamwinnum['CTCFbound'] = ch38drop['CTCFbound']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='CTCFbound', palette="tab10")
plt.title('Upstream principalupstreamwinnum with ch38drop class CTCFbound')
plt.show()


# # 10 most contributing 

# // upstream

# In[81]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

mostimportant10= most_important10_differentmethod[:,0]
mostimportant10= list(mostimportant10) 


# In[82]:


most_important10_differentmethod


# In[83]:



Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns 


# In[84]:



up_mostimportantsubset=Upstreamdropwin[mostimportant10]
up_mostimportantsubset


# In[85]:


importantupresult = pd.concat([up_mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']


# In[86]:



from sklearn.decomposition import PCA
pcadfupstreamMatreal = PCA(n_components=2)
principalComponentsdfupstreamMatreal = pcadfupstreamMatreal.fit_transform(Upstreamdropwin)

plt.bar(x=range(2), height= pcadfupstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfupstreamMatreal.explained_variance_ratio_)

principalupstreamDfreal = pd.DataFrame (data = principalComponentsdfupstreamMatreal, columns = ['a', 'b'])
plt.scatter(principalupstreamDfreal['a'], principalupstreamDfreal['b'], c='purple')
plt.title('most important 10 for upstream full Matrix')
plt.savefig('upstream_mostimportant10.pdf', dpi=299)
plt.show()


# In[87]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

down_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

down_mostimportant10= down_most_important10_differentmethod[:,0]
down_mostimportant10= list(down_mostimportant10)

down_most_important10_differentmethod
Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 
down_mostimportantsubset=Downstreamdropwin[down_mostimportant10]
down_mostimportantsubset

from sklearn.decomposition import PCA
pcadfdownstreamMatreal = PCA(n_components=2)
principalComponentsdfdownstreamMatreal = pcadfdownstreamMatreal.fit_transform(Downstreamdropwin)

plt.bar(x=range(2), height= pcadfdownstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfdownstreamMatreal.explained_variance_ratio_)

principaldownstreamDfreal = pd.DataFrame (data = principalComponentsdfdownstreamMatreal, columns = ['a', 'b'])
plt.scatter(principaldownstreamDfreal['a'], principaldownstreamDfreal['b'], c='purple')
plt.title('most important 10 for downstream full Matrix')
plt.savefig('downstream_mostimportant10_ch9.pdf', dpi=299)
plt.show()

from sklearn.decomposition import PCA
pcadfdownstreamMatreal = PCA(n_components=2)
principalComponentsdfdownstreamMatreal = pcadfdownstreamMatreal.fit_transform(Downstreamdropwin)

plt.bar(x=range(2), height= pcadfdownstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfdownstreamMatreal.explained_variance_ratio_)

principaldownstreamDfreal = pd.DataFrame (data = principalComponentsdfdownstreamMatreal, columns = ['a', 'b'])
plt.scatter(principaldownstreamDfreal['a'], principalupstreamDfreal['b'], c='purple')
plt.title('most important 10 for upstream full Matrix')
plt.savefig('upstream_mostimportant10_ch9.pdf', dpi=299)
plt.show()


# In[88]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principalupstreamwinnum[['mostimportant_1','most_important_2','mostimportant_3','mostimportant_4','mostimportant_5','mostimportant_6','mostimportant_7','mostimportant_8','mostimportant_9','mostimportant_10']] = up_mostimportantsubset

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue="mostimportant_1")


# In[ ]:


from sklearn.decomposition import PCA
pcadfupstreamMatreal = PCA(n_components=2)
principalComponentsdfupstreamMatreal = pcadfupstreamMatreal.fit_transform(Upstreamdropwin)

plt.bar(x=range(2), height= pcadfupstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfupstreamMatreal.explained_variance_ratio_)

principalupstreamDfreal = pd.DataFrame (data = principalComponentsdfupstreamMatreal, columns = ['a', 'b'])
plt.scatter(principalupstreamDfreal['a'], principalupstreamDfreal['b'], c='purple')
plt.title('most important 10 for upstream full Matrix')
plt.savefig('upstream_mostimportant10_chr930_5kb.pdf', dpi=299)
plt.show()


# // downstream

# In[ ]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Downstreamdropwin)

down_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

down_mostimportant10= down_most_important10_differentmethod[:,0]
down_mostimportant10= list(down_mostimportant10)


# In[ ]:


down_most_important10_differentmethod
Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 
down_mostimportantsubset=Downstreamdropwin[down_mostimportant10]
down_mostimportantsubset


# In[ ]:


from sklearn.decomposition import PCA
pcadfdownstreamMatreal = PCA(n_components=2)
principalComponentsdfdownstreamMatreal = pcadfdownstreamMatreal.fit_transform(Downstreamdropwin)

plt.bar(x=range(2), height= pcadfdownstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfdownstreamMatreal.explained_variance_ratio_)

principaldownstreamDfreal = pd.DataFrame (data = principalComponentsdfdownstreamMatreal, columns = ['a', 'b'])
plt.scatter(principaldownstreamDfreal['a'], principaldownstreamDfreal['b'], c='purple')
plt.title('most important 10 for downstream full Matrix')
plt.savefig('downstream_mostimportant10_ch9.pdf', dpi=299)
plt.show()


# In[ ]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principaldownstreamDfreal[['mostimportant_1','most_important_2','mostimportant_3','mostimportant_4','mostimportant_5','mostimportant_6','mostimportant_7','mostimportant_8','mostimportant_9','mostimportant_10']] = down_mostimportantsubset

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamDfreal, x="a", y="b", hue="mostimportant_1")


# // heatmap upstream

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:



# just correlationship
# upstream 
up_mostimportantsubset.corr()
plt.title('most important 10 for upstream full Matrix_ correlation_chr930_5kb')
plt.savefig('upstream_mostimportant10_corr_chr930_5kb.pdf', dpi=299)
sns.heatmap(up_mostimportantsubset.corr());

# downstream 
down_mostimportantsubset.corr()
plt.title('most important 10 for downstream full Matrix_ correlation_chr930_5kb')
plt.savefig('downstream_mostimportant10_corr_chr930_5kb.pdf', dpi=299)
sns.heatmap(down_mostimportantsubset.corr());


# # Extracting the most impotant 10: After concatenating the prediction the full matrix

# In[ ]:


upresult= upresult.rename(columns ={"up_prediction": "1083"})


# In[ ]:


downresult= downresult.rename(columns ={"down_prediction": "1083"})


# // upstream

# In[ ]:


upresult.columns = upresult.columns.astype(int) 
upresult.columns 

#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(upresult)

up_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

up_mostimportant10= up_most_important10_differentmethod[:,0]
up_mostimportant10= list(up_mostimportant10)

mostimportantsubset=Upstreamdropwin[mostimportant10]
mostimportantsubset

importantupresult = pd.concat([mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']


# // downstream

# In[ ]:


down_mostimportantsubset=Downstreamdropwin[down_mostimportant10]
down_mostimportantsubset

importantdownresult = pd.concat([down_mostimportantsubset, dfdownprediction], axis=1)
importantdownresult.columns = [*importantdownresult .columns[:-1], 'p']

importantdownresult


# In[ ]:



mostimportantsubset=Upstreamdropwin[mostimportant10]
print(mostimportantsubset)

importantupresult = pd.concat([mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']

importantupresult
meanupimportant = importantupresult.groupby('p').mean()
meanupdf=pd.DataFrame(meanupimportant)

print(meanupdf)


# In[ ]:


print(meanupdf)


# # groupby mean() and HeatMap

# In[ ]:




meandownimportant = importantdownresult.groupby('p').mean()
meandowndf=pd.DataFrame(meandownimportant)
meandowndf


countdown = importantdownresult.value_counts('p')
countup =importantupresult.value_counts('p')
dfcountup=pd.DataFrame(countup)
dfcountdown=pd.DataFrame(countdown)

dfcountupwhere = np.where((dfcountup[0]<10),False,True)
dfcountdownwhere = np.where((dfcountdown[0]<10),False,True)


# In[ ]:


dfcountup


# In[ ]:


dfcountdown


# In[ ]:


print(dfcountup[0]<10)


# In[ ]:


print(dfcountdown[0]<10)


# In[ ]:


meandowndf


# In[ ]:


meanupdf


# In[ ]:



upheatmapdf = meanupdf[dfcountupwhere]
upheatmapdf


# In[ ]:



downheatmapdf = meandowndf[dfcountdownwhere]
downheatmapdf


# In[ ]:


#upstream heatmap data
import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('most important 10 for upstream full Matrix_HeatMap_chr930_5kb')
plt.savefig('upstream_mostimportant10_HeatMap_ch9.pdf', dpi=299)
sns.heatmap(upheatmapdf, cmap ='RdYlGn', linewidths = 0.30, annot = True)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('most important 10 for downstream full Matrix_HeatMap_chr930_5kb')
plt.savefig('downstream_mostimportant10_HeatMap_chr930_5kb.pdf', dpi=299)
sns.heatmap(downheatmapdf, cmap ='RdYlGn', linewidths = 0.30, annot = True)


# In[ ]:


plt.title('most important 10 for downstream full Matrix_ClusterMap_chr930_5kb')
sns.clustermap(downheatmapdf)


plt.title('most important 10 for upstream full Matrix_ClusterMap_chr930_5kb')
sns.clustermap(upheatmapdf)


# # HeatMap and SeqClass (subset by SUMTR) on most important 10

# In[ ]:


def sum_df_down(df, tr_id, seqclass):
    #print (df)
    dfDownwinnum = df.loc[df["Result"] == (tr_id)].copy()
    sumTRDown =dfDownwinnum[str(9+seqclass)].sum()
    
    return sumTRDown



colsDown = list()
colsDown = dfDownwinnum.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRDown= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfDownwinnum["Result"].tolist())):
    for key in translater.keys():
        sumTRDown [key].append(sum_df_down(dfDownwinnum,i, translater[key]))

dfDownwinnumdrop = dfDownwinnum


def sum_df_up(df, tr_id, seqclass):
    #print (df)
    dfUpwinnum = df.loc[df["Result"] == (tr_id)].copy()
    sumTRUP =dfUpwinnum[str(9+seqclass)].sum()
    
    return sumTRUP

colsUP = list()
colsUP = dfUpwinnum.columns.tolist()
print(type(colsUP))
print(len(colsUP))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRUP= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfUpwinnum["Result"].tolist())):
    for key in translater.keys():
        sumTRUP[key].append(sum_df_up(dfUpwinnum,i, translater[key]))

dfUpwinnumdrop = dfUpwinnum


# // upstream

# In[ ]:


#pca.component , upstream first 

sumTRUPdf = pd.DataFrame(sumTRUP)
sumTRUPdfre=sumTRUPdf.rename(columns = {'PC1' :'1', 'E3': '2',	'E4': '3',	'HET1': '4', 'E8': '5', 'HET2' : '6', 'E9' : '7', 'HET3' :'8', 'PC4' :'9' , 'P' : '10',	'CTCF' : '11', 'E10': '12', 'HET4' : '13'}) 
sumTRUPfre = np.log10(sumTRUPdfre)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
x = StandardScaler().fit_transform(sumTRUPdfre)
df_pca = pca.fit_transform(x)

up_sum_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

up_sum_mostimportant10= up_sum_most_important10_differentmethod[:,0]
up_sum_mostimportant10= list(up_sum_mostimportant10)


# In[ ]:


sumTRUPdfre


# In[ ]:


sumTRUPdfre.columns = sumTRUPdfre.columns.astype(int) 
sumTRUPdfre.columns 
up_sum_mostimportantsubset=sumTRUPdfre[up_mostimportant10]
print(up_sum_mostimportantsubset)

from sklearn.decomposition import PCA
pcadfupsumstreamMatreal = PCA(n_components=2)
ComponentpcadfupsumstreamMatreal = pcadfupsumstreamMatreal.fit_transform(sumTRUPdfre)

plt.bar(x=range(2), height= pcadfupsumstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfupsumstreamMatreal.explained_variance_ratio_)

principalupDfsum = pd.DataFrame (data = ComponentpcadfupsumstreamMatreal, columns = ['a', 'b'])
plt.scatter(principalupDfsum ['a'], principalupDfsum ['b'], c='purple')
plt.show()


# In[ ]:



import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principalupDfsum[['mostimportant_1','most_important_2','mostimportant_3','mostimportant_4','mostimportant_5','mostimportant_6','mostimportant_7','mostimportant_8','mostimportant_9','mostimportant_10']] = mostimportantsubset

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupDfsum, x="a", y="b", hue="mostimportant_1")


# In[ ]:



importantsumtrup = pd.concat([sumTRUPdfre, dfupprediction], axis=1)
importantsumtrup.columns = [*importantsumtrup .columns[:-1], 'p']


meanupsumimportant = importantsumtrup.groupby('p').mean()


meanupsumdf=pd.DataFrame(meanupsumimportant)
meanupsumdf


upheatsumdf = meanupsumdf[dfcountupwhere]
upheatsumdf


# In[ ]:


#'PC1' :'1', 'E3': '2',	'E4': '3',	'HET1': '4', 'E8': '5', 
# 'HET2' : '6', 'E9' : '7', 'HET3' :'8', 'PC4' :'9' , 'P' : '10',	
# 'CTCF' : '11', 'E10': '12', 'HET4' : '13'


# In[ ]:


import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('most important 10 for upstream Seqclassdf_HeatMap_chr930_5kb')
plt.savefig('upstream_mostimportant10_HeatMap_seqclass_chr930_5kb.pdf', dpi=299)
sns.heatmap(upheatsumdf, cmap ='RdYlGn', linewidths = 0.30, annot = True)


# In[ ]:


plt.title('most important 10 for upstream Seqclassdf_ClusterMap_chr930_5kb')
sns.clustermap(upheatsumdf)


# // downstream

# In[ ]:


sumTRDowndf = pd.DataFrame(sumTRDown)
sumTRDowndfre=sumTRDowndf.rename(columns = {'PC1' :'1', 'E3': '2',	'E4': '3',	'HET1': '4', 'E8': '5', 'HET2' : '6', 'E9' : '7', 'HET3' :'8', 'PC4' :'9' , 'P' : '10',	'CTCF' : '11', 'E10': '12', 'HET4' : '13'}) 
sumTRDownfre = np.log10(sumTRDowndfre)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
x = StandardScaler().fit_transform(sumTRDowndfre)
df_pca = pca.fit_transform(x)

down_sum_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

down_sum_mostimportant10= down_sum_most_important10_differentmethod[:,0]
down_sum_mostimportant10= list(down_sum_mostimportant10)


# In[ ]:



import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt





importantsumtrdown = pd.concat([sumTRDowndfre, dfdownprediction], axis=1)
importantsumtrdown.columns = [*importantsumtrdown .columns[:-1], 'p']
importantsumtrdown
meandownsumimportant = importantsumtrdown.groupby('p').mean()
meandownsumdf=pd.DataFrame(meandownsumimportant)
meandownsumdf
downheatsumdf = meandownsumdf[dfcountdownwhere]
downheatsumdf


import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(downheatsumdf, cmap ='RdYlGn', linewidths = 0.30, annot = True)
plt.title('most important 10 for downstream Seqclassdf_HeatMap_chr930_5kb')
plt.savefig('downstream_mostimportant10_HeatMap_seqclass_chr930_5kb.pdf', dpi=299)
sns.clustermap(downheatsumdf)


# # Stastistical Analysis 
# // getting zscore 

# In[ ]:


import pandas as pd

dfsumTRpltdown = np.log10(dfsumTRpltdown)
dfsumTRpltdown
dfsumTRpltup = np.log10(dfsumTRpltup)
dfsumTRpltup


# In[ ]:


# upstream

# upstream with 60 seqclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA()
x = StandardScaler().fit_transform(dfsumTRpltup)
df_pca = pca.fit_transform(x)

up60_most_important10_differentmethod=np.abs(pca.components_)[0,:].argsort()[::-1][:10]

up60_sum_mostimportant10= up60_most_important10_differentmethod
up60_sum_mostimportant10= list(up60_sum_mostimportant10)

dfsumTRpltup.columns = dfsumTRpltup.columns.astype(int) 
dfsumTRpltup.columns 
up60_sum_mostimportantsubset=dfsumTRpltup[up60_sum_mostimportant10]
print(up60_sum_mostimportantsubset)

from sklearn.decomposition import PCA
pcadfupstream60 = PCA(n_components=2)
Componentpcadfupstream60 = pcadfupstream60.fit_transform(dfsumTRpltup)

plt.bar(x=range(2), height= pcadfupstream60.explained_variance_ratio_)

plt.show()

sum(pcadfupstream60.explained_variance_ratio_)

principalupDf60 = pd.DataFrame (data = Componentpcadfupstream60, columns = ['a', 'b'])
plt.scatter(principalupDf60  ['a'], principalupDf60  ['b'], c='purple')
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principalupDf60 [['mostimportant_1','most_important_2','mostimportant_3','mostimportant_4','mostimportant_5','mostimportant_6','mostimportant_7','mostimportant_8','mostimportant_9','mostimportant_10']] = up60_sum_mostimportant10

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupDf60, x="a", y="b", hue="mostimportant_1")
importantsumtr60up = pd.concat([dfsumTRpltup, dfupprediction], axis=1)
importantsumtr60up.columns = [*importantsumtr60up .columns[:-1], 'p']
meanupsumup60important = importantsumtr60up.groupby('p').mean()
meanupsum60=pd.DataFrame(meanupsumup60important)
meanupsum60
upheatsum60 = meanupsum60[dfcountupwhere]
upheatsum60
import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(upheatsum60, cmap ='RdYlGn', linewidths = 0.30, annot = True)
plt.title('most important 10 for upstream 60Seqclassdf_HeatMap_chr930_5kb')
plt.savefig('upstream_mostimportant10_HeatMap_60seqclass_chr930_5kb.pdf', dpi=299)
sns.clustermap(upheatsum60)


# In[ ]:


# downstream

# downstream with 60 seqclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
x = StandardScaler().fit_transform(dfsumTRpltdown)
df_pca = pca.fit_transform(x)

down60_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

down60_sum_mostimportant10= down60_most_important10_differentmethod[:,0]
down60_sum_mostimportant10= list(down60_sum_mostimportant10)

dfsumTRpltdown.columns = dfsumTRpltdown.columns.astype(int) 
dfsumTRpltdown.columns 
down60_sum_mostimportantsubset=dfsumTRpltdown[down60_sum_mostimportant10]
print(down60_sum_mostimportantsubset)

from sklearn.decomposition import PCA
pcadfdownstream60 = PCA(n_components=2)
Componentpcadfdownstream60 = pcadfdownstream60.fit_transform(dfsumTRpltdown)

plt.bar(x=range(2), height= pcadfdownstream60.explained_variance_ratio_)

plt.show()

sum(pcadfdownstream60.explained_variance_ratio_)

principaldownDf60 = pd.DataFrame (data = Componentpcadfdownstream60, columns = ['a', 'b'])
plt.scatter(principaldownDf60  ['a'], principaldownDf60  ['b'], c='purple')
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principaldownDf60 [['mostimportant_1','most_important_2','mostimportant_3','mostimportant_4','mostimportant_5','mostimportant_6','mostimportant_7','mostimportant_8','mostimportant_9','mostimportant_10']] = down60_sum_mostimportant10

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownDf60, x="a", y="b", hue="mostimportant_1")
importantsumtr60down = pd.concat([dfsumTRpltdown, dfdownprediction], axis=1)
importantsumtr60down.columns = [*importantsumtr60down .columns[:-1], 'p']
meandownsumdown60important = importantsumtr60down.groupby('p').mean()
meandownsum60=pd.DataFrame(meandownsumdown60important)
meandownsum60
downheatsum60 = meandownsum60[dfcountdownwhere]
downheatsum60
import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(downheatsum60, cmap ='RdYlGn', linewidths = 0.30, annot = True)
plt.title('most important 10 for downstream 60Seqclassdf_HeatMap_chr930_5kb')
plt.savefig('downstream_mostimportant10_HeatMap_60seqclass_chr930_5kb.pdf', dpi=299)
sns.clustermap(downheatsum60)


# # Zcore and heatmap

# In[ ]:


downpredictionbywdf=pd.DataFrame(dfdownpredictionbyw)
uppredictionbywdf=pd.DataFrame(dfuppredictionbyw)


# // upstream

# In[ ]:


df = pd.DataFrame(np.random.randint(100, 200, size=(5, 3)), columns=['A', 'B', 'C'])
df


# In[ ]:


from scipy.stats import zscore
zscoredfup = dfsumTRpltup.apply(zscore)

importantzscoreup = pd.concat([zscoredfup, uppredictionbywdf], axis=1)
importantzscoreup.columns = [*importantzscoreup.columns[:-1], 'p']
meanimportantzscoreup = importantzscoreup.groupby('p').mean()
meanimportantzscoreup60=pd.DataFrame(meanimportantzscoreup)
meanimportantzscoreup60
#zscoreup60heatmap = meanimportantzscoreup60[dfcountupwhere]


# In[ ]:


import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('upstream zscore_HeatMap_chr930_5kb')
plt.savefig('upstream_zscore__heatmap_chr930_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoreup60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreup60)


# // downstream

# In[ ]:


from scipy.stats import zscore
zscoredfdown = dfsumTRpltdown.apply(zscore)

importantzscoredown = pd.concat([zscoredfdown, downpredictionbywdf], axis=1)
importantzscoredown.columns = [*importantzscoredown.columns[:-1], 'p']
meanimportantzscoredown = importantzscoredown.groupby('p').mean()
meanimportantzscoredown60=pd.DataFrame(meanimportantzscoredown)
meanimportantzscoredown60
# zscoredown60heatmap = meanimportantzscoredown60[dfcountdownwhere]

import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('downstream zscore_HeatMa_chr930_5kb')
plt.savefig('downstream_zscore__heatmap.pdf_chr930_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60)


# # Louvain and reduced Umap

# In[ ]:


reductupumap2d_chr9['p'] = dfuppredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='p',palette="tab10")
plt.show()


# In[ ]:


reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


# # trace-back : window number 
# // indices 
# // seq class number 
# // ex) label them promoter window 1, 2, 3 
# // make the mapping for them 
# // scores 
# // automatically 

# # SumTR Revision to including all 60 seq Classes 
# 
# 

# In[ ]:


# upstream with 60 seqclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA()
x = StandardScaler().fit_transform(dfsumTRpltup)
df_pca = pca.fit_transform(x)

up60_most_important10_differentmethod=np.abs(pca.components_)[0,:].argsort()[::-1][:10]

up60_sum_mostimportant10= up60_most_important10_differentmethod
up60_sum_mostimportant10= list(up60_sum_mostimportant10)

dfsumTRpltup.columns = dfsumTRpltup.columns.astype(int) 
dfsumTRpltup.columns 
up60_sum_mostimportantsubset=dfsumTRpltup[up60_sum_mostimportant10]
print(up60_sum_mostimportantsubset)

from sklearn.decomposition import PCA
pcadfupstream60 = PCA(n_components=2)
Componentpcadfupstream60 = pcadfupstream60.fit_transform(dfsumTRpltup)

plt.bar(x=range(2), height= pcadfupstream60.explained_variance_ratio_)

plt.show()

sum(pcadfupstream60.explained_variance_ratio_)

principalupDf60 = pd.DataFrame (data = Componentpcadfupstream60, columns = ['a', 'b'])
plt.scatter(principalupDf60  ['a'], principalupDf60  ['b'], c='purple')
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principalupDf60 [['mostimportant_1','most_important_2','mostimportant_3','mostimportant_4','mostimportant_5','mostimportant_6','mostimportant_7','mostimportant_8','mostimportant_9','mostimportant_10']] = up60_sum_mostimportant10

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupDf60, x="a", y="b", hue="mostimportant_1")
importantsumtr60up = pd.concat([dfsumTRpltup, dfupprediction], axis=1)
importantsumtr60up.columns = [*importantsumtr60up .columns[:-1], 'p']
meanupsumup60important = importantsumtr60up.groupby('p').mean()
meanupsum60=pd.DataFrame(meanupsumup60important)
meanupsum60
upheatsum60 = meanupsum60[dfcountupwhere]
upheatsum60
import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(upheatsum60, cmap ='RdYlGn', linewidths = 0.30, annot = True)
plt.title('most important 10 for upstream 60Seqclassdf_HeatMap')
plt.savefig('upstream_mostimportant10_HeatMap_60seqclass_chr930_5kb.pdf', dpi=299)
sns.clustermap(upheatsum60)


# In[ ]:





# In[ ]:


# downstream with 60 seqclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
x = StandardScaler().fit_transform(dfsumTRpltdown)
df_pca = pca.fit_transform(x)

down60_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

down60_sum_mostimportant10= down60_most_important10_differentmethod[:,0]
down60_sum_mostimportant10= list(down60_sum_mostimportant10)

dfsumTRpltdown.columns = dfsumTRpltdown.columns.astype(int) 
dfsumTRpltdown.columns 
down60_sum_mostimportantsubset=dfsumTRpltdown[down60_sum_mostimportant10]
print(down60_sum_mostimportantsubset)

from sklearn.decomposition import PCA
pcadfdownstream60 = PCA(n_components=2)
Componentpcadfdownstream60 = pcadfdownstream60.fit_transform(dfsumTRpltdown)

plt.bar(x=range(2), height= pcadfdownstream60.explained_variance_ratio_)

plt.show()

sum(pcadfdownstream60.explained_variance_ratio_)

principaldownDf60 = pd.DataFrame (data = Componentpcadfdownstream60, columns = ['a', 'b'])
plt.scatter(principaldownDf60  ['a'], principaldownDf60  ['b'], c='purple')
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principaldownDf60 [['mostimportant_1','most_important_2','mostimportant_3','mostimportant_4','mostimportant_5','mostimportant_6','mostimportant_7','mostimportant_8','mostimportant_9','mostimportant_10']] = down60_sum_mostimportant10

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownDf60, x="a", y="b", hue="mostimportant_1")
importantsumtr60down = pd.concat([dfsumTRpltdown, dfdownprediction], axis=1)
importantsumtr60down.columns = [*importantsumtr60down .columns[:-1], 'p']
meandownsumdown60important = importantsumtr60down.groupby('p').mean()
meandownsum60=pd.DataFrame(meandownsumdown60important)
meandownsum60
downheatsum60 = meandownsum60[dfcountdownwhere]
downheatsum60
import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(downheatsum60, cmap ='RdYlGn', linewidths = 0.30, annot = True)
plt.title('most important 10 for downstream 60Seqclassdf_HeatMap')
plt.savefig('downstream_mostimportant10_HeatMap_60seqclass_chr930_5kb.pdf', dpi=299)
sns.clustermap(downheatsum60)


# In[ ]:


from scipy.stats import zscore
zscoredfup = dfsumTRpltup.apply(zscore)

importantzscoreup = pd.concat([zscoredfup, dfuppredictionbyw], axis=1)
importantzscoreup.columns = [*importantzscoreup.columns[:-1], 'p']
meanimportantzscoreup = importantzscoreup.groupby('p').mean()
meanimportantzscoreup60=pd.DataFrame(meanimportantzscoreup)
meanimportantzscoreup60
#zscoreup60heatmap = meanimportantzscoreup60[dfcountupwhere]

import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('upstream zscore_HeatMap_chr930_5kb')
plt.savefig('upstream_zscore__heatmap_chr930_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoreup60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
reductupumap2d_chr9['p'] = dfuppredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='p',palette="tab10")
plt.show()


# In[ ]:


from scipy.stats import zscore
zscoredfdown = dfsumTRpltdown.apply(zscore)

importantzscoredown = pd.concat([zscoredfdown, dfdownpredictionbyw], axis=1)
importantzscoredown.columns = [*importantzscoredown.columns[:-1], 'p']
meanimportantzscoredown = importantzscoredown.groupby('p').mean()
meanimportantzscoredown60=pd.DataFrame(meanimportantzscoredown)
meanimportantzscoredown60
# zscoredown60heatmap = meanimportantzscoredown60[dfcountdownwhere]

import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('downstream zscore_HeatMap_chr930_5kb')
plt.savefig('downstream_zscore__heatmap_chr930_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60)
reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


# In[ ]:


dfDownseqname = dfDownwinnum.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6','49':'losig1','50':'losig2','51':'losig3','52':'losig4','53':'losig5','54':'losig6','55':'losig7','56':'losig8','57':'losig9','58':'losig10','59':'losig11'})
dfDownseqname


# In[ ]:


from scipy.stats import zscore
zscoredfdownseq = dfDownseqname.apply(zscore)

importantzscoredownseq = pd.concat([zscoredfdownseq, dfdownpredictionbyw], axis=1)
importantzscoredownseq.columns = [*importantzscoredownseq.columns[:-1], 'p']
meanimportantzscoredownseq = importantzscoredownseq.groupby('p').mean()
meanimportantzscoredown60seq=pd.DataFrame(meanimportantzscoredownseq)
meanimportantzscoredown60seq
# zscoredown60heatmap = meanimportantzscoredown60[dfcountdownwhere]

import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('downstream zscore_HeatMap_chr930_5kb')
plt.savefig('downstream_zscore__heatmap_chr930_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60seq)
reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


# In[ ]:


dfUpseqname = dfUpwinnum.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6','49':'losig1','50':'losig2','51':'losig3','52':'losig4'})
dfUpseqname


# In[ ]:


dfUpseqname=dfUpseqname.drop(columns=['TR_id', 'Win_num','Result'])


# In[ ]:



from scipy.stats import zscore
zscoredfupseq = dfUpseqname.apply(zscore)

importantzscoredupseq = pd.concat([zscoredfupseq, dfuppredictionbyw], axis=1)
importantzscoredupseq.columns = [*importantzscoredupseq.columns[:-1], 'p']
meanimportantzscoredupseq = importantzscoredupseq.groupby('p').mean()
meanimportantzscoredup60seq=pd.DataFrame(meanimportantzscoredupseq)
meanimportantzscoredup60seq
# zscoredown60heatmap = meanimportantzscoredown60[dfcountdownwhere]

import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('upstream zscore_HeatMap_chr930_5')
plt.savefig('upstream_zscore__heatmap_chr930_5.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredup60seq)


# In[ ]:


dfDownseqname = dfDownwinnum.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6','49':'losig1','50':'losig2','51':'losig3','52':'losig4'})
dfDownseqname
dfDownseqname= dfDownseqname.drop(columns=['TR_id', 'Win_num','Result'])
from scipy.stats import zscore
zscoredfdownseq = dfDownseqname.apply(zscore)

importantzscoreddownseq = pd.concat([zscoredfdownseq, dfdownpredictionbyw], axis=1)
importantzscoreddownseq.columns = [*importantzscoreddownseq.columns[:-1], 'p']
meanimportantzscoreddownseq = importantzscoreddownseq.groupby('p').mean()
meanimportantzscoreddown60seq=pd.DataFrame(meanimportantzscoreddownseq)
meanimportantzscoreddown60seq
# zscoredown60heatmap = meanimportantzscoredown60[dfcountdownwhere]

import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('downstream zscore_HeatMap_chr930_5')
plt.savefig('downstream_zscore__heatmap_chr930_5.pdf', dpi=299)
sns.heatmap(meanimportantzscoreddown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreddown60seq)


# In[ ]:


# all the needed import untill ouvain 
import numpy as np
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('conda install -c conda-forge umap-learn -y')

# Data manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Visualization
import plotly.express as px # for data visualization
import matplotlib.pyplot as plt # for showing handwritten digits

# Skleran
from sklearn.datasets import load_digits # for MNIST data
from sklearn.model_selection import train_test_split # for splitting data into train and test samples

# UMAP dimensionality reduction
from umap import UMAP
import umap
get_ipython().system('pip install umap-learn')

from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn

from sklearn.manifold import TSNE

from umap import UMAP
import umap
get_ipython().system('pip install umap-learn')

from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn

from sklearn.manifold import TSNE

# tsne
X_2u2=  pd.read_csv("X_2u2_chr930_5kb.csv")
X_2u1= pd.read_csv("X_2u1_chr930_5kb.csv")
X_2u= pd.read_csv("X_2u_chr930_5kb.csv")
X_2d= pd.read_csv("X_2d_chr930_5kb.csv")
X_2d1= pd.read_csv("X_2d1_chr930_5kb.csv")
X_2d2= pd.read_csv("X_2d2_chr930_5kb.csv")
# pca
principalupstreamwinnum= pd.read_csv("principalupstreamwinnum_chr930_5kb")
principaldownstreamwinnum= pd.read_csv("principaldownstreamwinnum_chr930_5kb")
#full matrix
dfDownwinnum= pd.read_csv("dfDownwinnum_chr930_5kb")
dfUpwinnum= pd.read_csv("dfUpwinnum_chr930_5kb")
Upstreamdropwin = pd.read_csv("Upstreamdropwin_chr930_5kb.csv")
Downstreamdropwin = pd.read_csv("Downstreamdropwin_chr930_5kb")
#louvain
upresult= pd.read_csv("upresult_chr930_5kb.tsv")
downresult= pd.read_csv("downresult_chr930_5kb.tsv")
dfupprediction= pd.read_csv("dfupprediction_chr930_5kb.tsv")
dfdownprediction= pd.read_csv("dfdownprediction_chr930_5kb.tsv")
# save the dfs
dfsumTRpltup= pd.read_csv("dfsumTRpltup_chr930_5kb.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chr930_5kb.tsv")
umap2dimensionup_chr9= pd.read_csv("umap2dimensionup_chr930_5kb.tsv")
umap2dimensiondown_chr9= pd.read_csv("umap2dimensiondown_chr930_5kb.tsv")
# reduced 
reductdownumap2d= pd.read_csv("reductdownumap2d_chr930_5kb.tsv")
reductupumap2d_chr9= pd.read_csv("reductupumap2d_chr930_5kb.tsv")
# louvain byw
dfdownpredictionbyw= pd.read_csv("downpredictionbyw_chr930_5kb.tsv")
dfuppredictionbyw= pd.read_csv("uppredictionbyw_chr930_5kb.tsv")


# In[ ]:


# labeldf with the umap


# In[ ]:


reductdownumap2d


# In[ ]:


reductdownumap2d


# In[ ]:


dfDownseqname


# In[ ]:





# In[ ]:


reductdownumap2d['CTCF'] = np.where((dfDownseqname['CTCF']<5000)&(dfDownseqname['CTCF']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x="a", y="b", hue='CTCF', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 5000')
plt.show()


# In[ ]:


print(DownstreamMatdropwinum)


# In[ ]:


print(UpstreamMatdropwinum)


# In[ ]:


DownstreamMatdropwinum = pd.DataFrame(DownstreamMatdropwinum)


# In[ ]:


UpstreamMatdropwinum = pd.DataFrame(UpstreamMatdropwinum)


# In[ ]:


dfsumTRpltup
dfsumTRpltdown


# In[ ]:


dfUpseqname = dfsumTRpltup.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfUpseqname


# In[ ]:



from scipy.stats import zscore
zscoredfupseq = dfUpseqname.apply(zscore)

importantzscoredupseq = pd.concat([zscoredfupseq, dfuppredictionbyw], axis=1)
importantzscoredupseq.columns = [*importantzscoredupseq.columns[:-1], 'p']
meanimportantzscoredupseq = importantzscoredupseq.groupby('p').mean()
meanimportantzscoredup60seq=pd.DataFrame(meanimportantzscoredupseq)
meanimportantzscoredup60seq
# zscoredown60heatmap = meanimportantzscoredown60[dfcountdownwhere]

import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(15.7,8.27)})
plt.title('upstream zscore_HeatMap_chr930_5')
plt.savefig('upstream_zscore__heatmap_chr930_5.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.set(rc={'figure.figsize':(15.7,8.27)})
sns.clustermap(meanimportantzscoredup60seq)
sns.set(rc={'figure.figsize':(15.7,8.27)})


# In[ ]:


sns.set(rc={'figure.figsize':(35.7,8.27)})
sns.clustermap(meanimportantzscoredup60seq,square=False,figsize=(15.7,8.27), xticklabels=1)


# In[ ]:


dfDownseqname = dfsumTRpltdown.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfDownseqname


# In[ ]:


from scipy.stats import zscore
zscoredfdownseq = dfDownseqname.apply(zscore)

importantzscoredownseq = pd.concat([zscoredfdownseq, dfdownpredictionbyw], axis=1)
importantzscoredownseq.columns = [*importantzscoredownseq.columns[:-1], 'p']
meanimportantzscoredownseq = importantzscoredownseq.groupby('p').mean()
meanimportantzscoredown60seq=pd.DataFrame(meanimportantzscoredownseq)
meanimportantzscoredown60seq
# zscoredown60heatmap = meanimportantzscoredown60[dfcountdownwhere]

import numpy as np 
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('downstream zscore_HeatMap_chr930_5kb')
plt.savefig('downstream_zscore__heatmap_chr930_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60seq, square=False,figsize=(15.7,8.27), xticklabels=1)


# In[ ]:


reductdownumap2d['CTCF'] = np.where((dfDownseqname['CTCF']<1000)&(dfDownseqname['CTCF']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 1000')
plt.show()


# In[ ]:


dfDownseqname


# In[ ]:


reductupumap2d_chr9


# In[ ]:


reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()


# In[ ]:


reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()


# In[ ]:


plt.hist(reductdownumap2d['p'])


# In[ ]:


plt.hist(reductdownumap2d['CTCF'])


# In[ ]:


reductupumap2d_chr9['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chr9['CTCF_high'] = np.where(reductupumap2d_chr9['CTCF']>10,True,False)
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='CTCF_high',palette="tab10")
plt.show()


# In[ ]:


reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


# In[ ]:





ch38drop['CTCFbound']= None
ch38drop

ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])


# In[ ]:


reductdownumap2d['CTCFbound'] = ch38drop['CTCFbound']


# In[ ]:


reductdownumap2d['CTCFbound'] = ch38drop['CTCFbound']
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCFbound',palette="tab10")
plt.show()


# In[ ]:


ch38drop['CTCFbound']


# In[ ]:


reductdownumap2d['distance'] = np.where((ch38drop['distance']<500)&(ch38drop['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='distance', palette="tab10")


plt.show()


# In[ ]:




