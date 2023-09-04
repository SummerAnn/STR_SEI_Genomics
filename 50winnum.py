#!/usr/bin/env python
# coding: utf-8

# # Input (VCF+SEI)

# In[ ]:





# 

# In[1]:


import numpy as np
import pandas as pd
combined4colbasedf = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/vcf/combined4colbase.vcf",sep="\t", header=None, names=['chrX','1','2','3','4'])
combined4colbasedf
df=pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/tsv/combined4colbase.ref_combined.tsv", sep="\t",low_memory=False)
df = df.drop_duplicates(subset=["1"])
print(df)

frames= [df, combined4colbasedf]
finalinput =pd.merge(right=combined4colbasedf, left=df, on=["1","2"])
dfinput= finalinput
display(dfinput)


dfinput.columns = [ "0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","chromosome","window","basepair"]
dfinput


display(dfinput)
dfinput = dfinput.join(dfinput['window'].str.split('_', expand=True).rename (columns={0:'TR_id', 1:'Win_num'}))
dfinput['Win_num']=dfinput['Win_num'].astype(int)
dfinput['TR_id']=dfinput['TR_id'].astype(int)
subset= dfinput[dfinput.Win_num<=50] 
subset= subset.sort_values(by=['TR_id','Win_num'])
subset

dfUpstreamdropwinnum = subset.drop_duplicates(subset=["TR_id","Win_num"], keep="first") 
print(dfUpstreamdropwinnum)

dfDownstreamdropwinnum = subset.drop_duplicates(subset=["TR_id","Win_num"], keep="last") 
print(dfDownstreamdropwinnum)


# # Adding the Column to iterate thourgh, for Matrix Flattening

# In[7]:


dfDownstreamdropwinnum=(dfDownstreamdropwinnum.reset_index(drop=True))
dfDownwinnum = dfDownstreamdropwinnum.drop(columns=['0', '1','2','3','4','5','6','7','8', 'chromosome','window','basepair'])


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


# In[8]:


dfUpstreamdropwinnum=(dfUpstreamdropwinnum.reset_index(drop=True))
dfUpwinnum = dfUpstreamdropwinnum.drop(columns=['0', '1','2','3','4','5','6','7','8', 'chromosome','window','basepair'])

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


# # Matrix Flattening 

# In[9]:




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
np.save("DownstreamMatwinum",DownstreamMatdropwinum)
print(DownstreamMatdropwinum)


# In[10]:




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
np.save("UpstreamMatwinum",UpstreamMatdropwinum)
print(UpstreamMatdropwinum)


# In[11]:


log10downstreamdropwin = np.log10(DownstreamMatdropwinum)
print(log10downstreamdropwin)


# In[12]:


log10upstreamdropwin = np.log10(UpstreamMatdropwinum)
print(log10upstreamdropwin)


# # DOWNSTREAM PCA

# In[13]:


from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Downstreamdropwin= pd.DataFrame(log10downstreamdropwin)
print(Downstreamdropwin)

Downstreamdropwin.to_csv("Downstreamdropwin", index= None)
Downstreamdropwin = pd.read_csv("Downstreamdropwin")

Downstreamdropwin
x = Downstreamdropwin.values
y = Downstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfdownwinnum = PCA(n_components=8)
pcadfdownwinnummatrix = pcadfdownwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfdownwinnum.explained_variance_ratio_)
plt.savefig('pcadownstreambar.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfdownwinnum.explained_variance_ratio_)


principaldownstreamwinnum = pd.DataFrame (data = pcadfdownwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principaldownstreamwinnum['a'], principaldownstreamwinnum['b'], c='green')

plt.savefig('pcadownstream.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()


# # UPSTREAM PCA

# In[14]:




Upstreamdropwin= pd.DataFrame(log10upstreamdropwin)
print(Upstreamdropwin)

Upstreamdropwin.to_csv("Upstreamdropwin", index= None)
Upstreamdropwin = pd.read_csv("Upstreamdropwin")

Upstreamdropwin
x = Upstreamdropwin.values
y = Upstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfupwinnum = PCA(n_components=8)
pcadfupwinnummatrix = pcadfupwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfupwinnum.explained_variance_ratio_)
plt.savefig('pcaupstreambar.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfupwinnum.explained_variance_ratio_)

principalupstreamwinnum = pd.DataFrame (data = pcadfupwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalupstreamwinnum['a'], principalupstreamwinnum['b'], c='green')
plt.savefig('pcaupstream.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()


# In[15]:


xup= StandardScaler().fit_transform(x)


# In[16]:


Downstreamdropwin


# In[17]:


Upstreamdropwin


# In[18]:


principalupstreamwinnum.to_csv("principalupstreamwinnum", index= None)
principaldownstreamwinnum.to_csv("principaldownstreamwinnum", index= None)


# # TSNE DownStream 

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
plt.savefig('tsnedownstream,perplexity:1000.pdf', dpi=299)

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
plt.savefig('tsnedownstream,perplexity:50.pdf', dpi=299)
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
plt.savefig('tsnedownstream,perplexity:5.pdf', dpi=299)
plt.show()


# In[20]:


X_2d 


# # TSNE Upstream

# In[21]:


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
plt.savefig('tsneupstream,perplexity:1000.pdf', dpi=299)
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
plt.savefig('tsneupstream,perplexity:50.pdf', dpi=299)
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
plt.savefig('tsneupstream,perplexity:5.pdf', dpi=299)
plt.show()


# In[22]:


X_2u2


# In[23]:


X_2u2.to_csv("X_2u2.csv", index= None)
X_2u1.to_csv("X_2u1.csv", index= None)
X_2u.to_csv("X_2u.csv", index= None)
X_2d.to_csv("X_2d.csv", index= None)
X_2d1.to_csv("X_2d1.csv", index= None)
X_2d2.to_csv("X_2d2.csv", index= None)


# # Upstream coloring (Not usable but keeping it as a reference)

# In[24]:


principalupstreamwinnum[['a','b','c','e','f','g','h']] 


## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue="b")


# # Downstream coloring (Not Quite Usable, but Keeping it as a reference)

# In[25]:


principaldownstreamwinnum[['a','b','c','e','f','g','h']] 


## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue="c")


# # DF Winnmum UP

# In[26]:


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


# In[27]:


for key in translater.keys():
    plt.hist(sumTRUP[key])
    plt.title("Bar, seqClass: "+ key + ", Upstream 5kb")
    plt.savefig("seqClass" + key+ "upstream.pdf",dpi=299)
    plt.show()


# In[28]:


for key in translater.keys():
    plt.plot(sumTRUP[key])
    plt.title(key)
    plt.savefig('upstream sumTR seqClass line', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()


# In[29]:


for key in translater.keys():

    X_2u1[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2u1, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Upstream 5kb")
    plt.savefig("seqClass" + key+ "upstream_tsne:5.pdf",dpi=299)
    plt.show()


# In[30]:


for key in translater.keys():

    X_2u2[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2u2, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 50, seqClass: "+ key + ", Upstream 5kb")
    plt.savefig("seqClass" + key+ "upstream_tsne:50.pdf",dpi=299)
    plt.show()
    
    


# # UPStream PCA coloring by SumTR

# In[31]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


## if you have any existing df and want to add columns (the same # of rows,take the data) = add as a column 
## 3 important, take the data 3 columns ( easy way to create the columns) 
principalupstreamwinnum[key] = sumTRUP[key]

## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue=key)


# In[32]:


dfUpwinnum


# In[33]:


dfUpwinnumdrop


# In[34]:


fig,ax = plt.subplots(-(-len(translater.keys())//3),3)

for i,key in enumerate(translater.keys()):   
    
    ax[i%4][i%2].hist(sumTRUP[key])
    ax[i%4][i%2].set_title(key)
    

    #yaxis: num TR xaxis: sumval
    #hist: how common are the values 


# In[35]:


for key in translater.keys():

    principalupstreamwinnum[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue=key)
    plt.title("pca, seqClass: "+ key + ", Upstream 5kb")
    plt.savefig("seqClass" + key+ "upstream_pca.pdf",dpi=299)
    plt.show()


# # Subset by more clustered region Tsne

# In[36]:


dfUpwinnum[dfUpwinnum.Result.isin(X_2d[X_2d.P > 750].index+1)]


# In[ ]:


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
Z= dfUpwinnum.values
X_2dd = tsne.fit_transform(Z)



X_2dd = pd.DataFrame (data = X_2d, columns = ['a', 'b'])
plt.scatter(X_2d ['a'], X_2d ['b'], c='green')
plt.title('With perplexity = 1000, tsne for dfUpwinnum')
plt.show()


n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=50,
              random_state=12)
Z= dfUpwinnum.values
X_2dd = tsne.fit_transform(Z)



X_2dd = pd.DataFrame (data = X_2dd, columns = ['a', 'b'])
plt.scatter(X_2dd ['a'], X_2dd ['b'], c='green')
plt.title('With perplexity = 50, tsne for dfUpwinnum')
plt.show()


# # DF Winnmum DOWN

# In[37]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfDownwinnum = df.loc[df["Result"] == (tr_id)].copy()
    sumTR =dfDownwinnum[str(9+seqclass)].sum()
    
    return sumTR



colsDown = list()
colsDown = dfDownwinnum.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTR= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfDownwinnum["Result"].tolist())):
    for key in translater.keys():
        sumTR[key].append(sum_df(dfDownwinnum,i, translater[key]))

dfDownwinnumdrop = dfDownwinnum


# In[38]:


for key in translater.keys():
    plt.plot(sumTR[key])
    plt.title(key)
    plt.show()


# In[39]:


for key in translater.keys():

    X_2d[key] = sumTR[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Downstream 5kb")
    plt.savefig("seqClass" + key+ "downstream_tsne:5.pdf",dpi=299)
    plt.show()


# # DOWNStream PCA coloring by SumTR

# In[40]:


for key in translater.keys():

    principaldownstreamwinnum[key] = sumTR[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue=key)
    plt.title("pca, seqClass: "+ key + ", Downstream 5kb")
    plt.savefig("seqClass" + key+ "downstream_pca.pdf",dpi=299)
    plt.show()


# In[41]:


for key in translater.keys():

    X_2d1[key] = sumTR[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d1, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 50, seqClass: "+ key + ", Downstream 5kb")
    plt.savefig("seqClass" + key+ "downstream_tsne:5.pdf",dpi=299)
    plt.show()


# In[42]:


for key in translater.keys():

    principaldownstreamwinnum[key] = sumTR[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue=key)
    plt.title(key)
    plt.show()


# # most Contirbuting 3 (UPstream)

# In[43]:


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


# In[44]:


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


# In[45]:


Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns 


# In[46]:


Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns 


# In[47]:


from sklearn.decomposition import PCA
pcadfupstreamMatreal = PCA(n_components=2)
principalComponentsdfupstreamMatreal = pcadfupstreamMatreal.fit_transform(Upstreamdropwin)

plt.bar(x=range(2), height= pcadfupstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfupstreamMatreal.explained_variance_ratio_)

principalupstreamDfreal = pd.DataFrame (data = principalComponentsdfupstreamMatreal, columns = ['a', 'b'])
plt.scatter(principalupstreamDfreal['a'], principalupstreamDfreal['b'], c='purple')
plt.show()


# In[48]:


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


# # most Contirbuting 3 (DOWNstream)

# In[49]:


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


# In[50]:


Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 


# In[51]:


mostimportantsubsetDown=Downstreamdropwin[mostimportant3ones]
mostimportantsubsetDown


# In[52]:


from sklearn.decomposition import PCA
pcadfdownstreamMatreal = PCA(n_components=2)
principalComponentsdfdownstreamMatreal = pcadfdownstreamMatreal.fit_transform(Downstreamdropwin)

plt.bar(x=range(2), height= pcadfdownstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfdownstreamMatreal.explained_variance_ratio_)

principaldownstreamDfreal = pd.DataFrame (data = principalComponentsdfdownstreamMatreal, columns = ['a', 'b'])
plt.scatter(principaldownstreamDfreal['a'], principaldownstreamDfreal['b'], c='green')
plt.show()


# In[53]:


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


# # DF winnum subset by SumTR data  (according to P) /Upstream

# In[54]:


SubdfUpwim=dfUpwinnum[dfUpwinnum.Result.isin(X_2d[X_2d.P > 750].index+1)]
SubdfUpwim


# In[55]:


newSubdfUpwim=(SubdfUpwim.reset_index(drop=True))


result = []
i = 0
for j in range(len(newSubdfUpwim["TR_id"])):
   
    
    if j == len(newSubdfUpwim["TR_id"])-1:
        result.append(i)
        
    elif newSubdfUpwim["TR_id"].iloc[j-1] != newSubdfUpwim["TR_id"].iloc[j]:
        result.append(i+1)
        i=i+1
          # if j ==0 append (i) 
    else:
        result.append(i)

newSubdfUpwim["Seq"] = result  
print(newSubdfUpwim)


# In[56]:


newSubdfUpwim.to_csv('newSubdfUpwim.tsv',sep='\t',index = False)

def condense_df(df, tr_id):
    #UpstreamMat = df.loc[df["TR_id"] == str(tr_id)].copy()
    newSubdfUpwinums= df.loc[df["Seq"] == (tr_id)].copy()
    newSubdfUpwinums.drop("Result", axis=1,inplace=True)
    newSubdfUpwinums.drop("TR_id", axis=1,inplace=True)
    newSubdfUpwinums.drop("Win_num", axis=1,inplace=True)
    newSubdfUpwinums.drop("Seq", axis=1,inplace=True)
    arrNew = newSubdfUpwinums.to_numpy().flatten(order='F')
    return arrNew


colsUPs = list()
colsUPs = newSubdfUpwim.columns.tolist()
#cols_newUP = [colsUP[-1]]
print(type(colsUPs))
print(len(colsUPs))
#cols_newUP.extend(colsUP[0:]) 
newSubdfUpwims=newSubdfUpwim
print(newSubdfUpwims.shape)
#print(UpstreamMatrix.columns)

newSubdfUpwinums = np.zeros(shape=(len(set(newSubdfUpwims["Seq"].tolist())), len(condense_df(newSubdfUpwims, 1))))

print(newSubdfUpwinums.shape)
cnt = 0
failed_ids= []

for i in list(set(newSubdfUpwims["Seq"].tolist())):
    cnt +=1
    try:
        newSubdfUpwinums[int (i)-1,:] = condense_df(newSubdfUpwims, i)
    except:
        failed_ids.append(i) 



newSubdfUpwims_copy = newSubdfUpwinums
failed_ids = [i-1 for i in list(map(int,failed_ids))]
newSubdfUpwinumms = np.delete (newSubdfUpwims_copy, failed_ids, axis = 0)
np.save("newSubdfUpwinums",newSubdfUpwinumms)
print(newSubdfUpwinumms)


# In[57]:


log10newSubdfUpwinumms = np.log10(newSubdfUpwinumms)
print(log10newSubdfUpwinumms)


# In[58]:


from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

log10newSubdfUpwinumms= pd.DataFrame(log10newSubdfUpwinumms)
print(log10newSubdfUpwinumms)

log10newSubdfUpwinumms.to_csv("log10newSubdfUpwinumms", index= None)
log10newSubdfUpwinumms = pd.read_csv("log10newSubdfUpwinumms")

log10newSubdfUpwinumms
x = log10newSubdfUpwinumms.values
y = log10newSubdfUpwinumms.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadflog10newSubdfUpwinumms = PCA(n_components=8)
pcadflog10newSubdfUpwinummsmatrix = pcadflog10newSubdfUpwinumms.fit_transform(x)

plt.bar(x=range(8), height= pcadflog10newSubdfUpwinumms.explained_variance_ratio_)

plt.show()

sum(pcadflog10newSubdfUpwinumms.explained_variance_ratio_)

principalpcadflog10newSubdfUpwinumms = pd.DataFrame (data = pcadflog10newSubdfUpwinummsmatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalpcadflog10newSubdfUpwinumms['a'], principalpcadflog10newSubdfUpwinumms['b'], c='green')
plt.show()


# In[59]:


for key in translater.keys():

    principalpcadflog10newSubdfUpwinumms[key] = sumTR[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalpcadflog10newSubdfUpwinumms, x="a", y="b", hue=key)
    plt.title(key)
    plt.show()
    


# In[ ]:


principalpcadflog10newSubdfUpwinumms


# In[ ]:


for key in translater.keys():

  principalpcadflog10newSubdfUpwinumms[key] = sumTR[key]
  sns.set(rc={'figure.figsize':(11.7,8.27)})
  sns.scatterplot(data=principalpcadflog10newSubdfUpwinumms, x="a", y="b", hue=key)
  plt.title("pca, seqClass: "+ key + ", Upstream 5kb-subset of sumTR according to Promoter")
  plt.savefig("seqClass" + key+ "upstream_pcasubsetbypromoter.pdf",dpi=299)
  plt.show()


# # Nearest Exon Coloring

# In[ ]:


# input 


# In[ ]:


nearExonTRID = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/sorteddropdup.bed", sep="\t", header=None,names=['1','2','TR_id'])
nearExonTRID

nearExon = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/exon/nearestExon.bed", sep="\t", header=None,names=['chrX','1','2','3','4','5','6'])
nearExon

dropExon = nearExon.drop_duplicates(subset=["2","3"],keep="first")
dropExon


frames= [dropExon, nearExonTRID]
finalExoninput =pd.merge(right=nearExonTRID, left=dropExon, on=["1","2"])

display(finalExoninput)


# In[ ]:


np.where(finalExoninput['6']>5000, True, False)


# # UPstream Exon PCA/TSNE data

# In[ ]:


plt.hist(finalExoninput['6'])


# In[ ]:



principalupstreamwinnum['exon'] = np.where((finalExoninput['6']>-5000)&(finalExoninput['6']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='exon')
plt.title("Nearest Exon, Upstream on pca")
plt.savefig("upstream_pca_nearestexon.pdf",dpi=299)


# # DOWNStream Exon PCA/TSNE data

# In[ ]:


principaldownstreamwinnum['exon'] = np.where((finalExoninput['6']>-5000)&(finalExoninput['6']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='exon')
plt.title("Nearest Exon, Downstream on pca")
plt.savefig("downstream_pca_nearestexon.pdf",dpi=299)


# In[ ]:


5


# In[ ]:


finalExoninput['6'].value_counts()


# # transcription start site Coloring

# In[ ]:


TSS_coord = pd.read_csv("/data/projects/nanopore/RepeatExpansion/coordinates/TSS_coords.bed",sep="\t", header=None, names=['chrX','0','1','2','3'])
TSS_coord


# In[ ]:


import numpy as np
import pandas as pd
TSS_merged = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/TSS/testTSSmerge.bed", sep="\t", header=None, names=['chrX','1','2','chrX1','4','5','6'])
TSS_merged 


# In[ ]:


dropTSS_merged=TSS_merged.drop_duplicates(subset=["1","2"],keep="first") 
dropTSS_merged


# # Upstream PCA/TSNE with TSS

# In[ ]:


X_2u['TSS'] = np.where((dropTSS_merged['6']>-5000)&(dropTSS_merged['6']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='TSS')
plt.title('Upstream TSNE with TSS')
plt.show()


# In[ ]:


X_2u1['TSS'] = np.where((dropTSS_merged['6']>-5000)&(dropTSS_merged['6']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u1, x="a", y="b", hue='TSS')
plt.title('Upstream TSNE with TSS')
plt.show()


# In[ ]:


X_2u2['TSS'] = np.where((dropTSS_merged['6']>-5000)&(dropTSS_merged['6']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u2, x="a", y="b", hue='TSS')
plt.title('Upstream TSNE with TSS')
plt.show()


# In[ ]:


principalupstreamwinnum['TSS'] = np.where((dropTSS_merged['6']>-5000)&(dropTSS_merged['6']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='TSS')
plt.title("TSS upstram")
plt.show()


# # Downstream PCA/TSNE with TSS data

# In[ ]:


principaldownstreamwinnum['TSS'] = np.where((dropTSS_merged['6']>-5000)&(dropTSS_merged['6']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='TSS')
plt.title("PCA and TSS DOWNSTREAM")
plt.show()


# In[ ]:


X_2d['TSS'] = np.where((dropTSS_merged['6']>-5000)&(dropTSS_merged['6']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='TSS')
plt.title("TSNE and TSS downstream")
plt.show()


# # CTCF Coloring

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


CTCF = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/CTCF/droppedchrXClosest.bed", sep="\t", header=None, names=['1','2','chrx','3','4','CTCF','closet'])
CTCF


# In[ ]:


CTCF_merged=CTCF.drop_duplicates(subset=["1","2"],keep="first") 
CTCF_merged


# In[ ]:


X_2u['CTCF'] = np.where((CTCF_merged['closet']>-5000)&(CTCF_merged['closet']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='CTCF')
plt.title('Upstream TSNE with CTCF')
plt.savefig("upstream_TSNE_CTCF.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d['CTCF'] = np.where((CTCF_merged['closet']>-5000)&(CTCF_merged['closet']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='CTCF')
plt.title('Downstream TSNE with CTCF')
plt.savefig("downstream_TSNE_CTCF.pdf",dpi=299)
plt.show()


# In[ ]:


X_2u2['CTCF'] = np.where((CTCF_merged['closet']>-5000)&(CTCF_merged['closet']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u2, x="a", y="b", hue='CTCF')
plt.title('Downstream TSNE:50 with CTCF')
plt.savefig("downstream_TSNE50_CTCF.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d2['CTCF'] = np.where((CTCF_merged['closet']>-5000)&(CTCF_merged['closet']<-0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d2, x="a", y="b", hue='CTCF')
plt.title('Downstream TSNE:50 with CTCF')
plt.savefig("downstream_TSNE:50_CTCF.pdf",dpi=299)
plt.show()


# # region length Coloring

# In[ ]:


regionlen = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/closetregionlen.bed", sep="\t", header=None, names=['chrx','1','2','chrx1','3','4','length','tr_id','null'])
regionlen


# In[ ]:


X_2d['length'] =regionlen['length']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='length')
plt.title("tsne for downstream, TSNE")
plt.show()
    


# In[ ]:


X_2u['length'] =regionlen['length']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='length')
plt.title("upstream tsne with Region Length")
plt.show()


# In[ ]:


X_2u2['length'] =regionlen['length']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u2, x="a", y="b", hue='length')
plt.title("upstream tsne with Region Length")
plt.show()


# In[ ]:


principaldownstreamwinnum['length'] =regionlen['length']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='length')
plt.title("downstream pca with Region Length")
plt.show()


# In[ ]:


principalupstreamwinnum['length'] =regionlen['length']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='length')
plt.title("upstream pca with Region length")
plt.show()


# # Repeat Masker/the repeatmasker classes corresponde to all transposon and retrotransposon classes

# In[ ]:


import numpy as np
import pandas as pd
repeatmasker = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/closestsortrepeatchrX.bed", sep="\t", header=None, names=['chrX','1','2','chrX1','3','4','class','distance'])
repeatmasker


# In[ ]:


droprepeatmasker=repeatmasker.drop_duplicates(subset=["1","2"],keep="first") 
droprepeatmasker


# In[ ]:


principalupstreamwinnum['distance'] =droprepeatmasker['distance']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance')
plt.title("upstream pca with repeatmasker by distance")
plt.show()


# # filter by distance (Repeat Masker)

# In[ ]:


X_2u['distance'] = np.where((droprepeatmasker['distance']>-2000)&(droprepeatmasker['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='distance')
plt.title('Upstream tsne:5 with RepeatMasker filter by distance>-2000')
plt.savefig("upstream_TSNE:5_RepeatMakser>-2000.pdf",dpi=299)
plt.show()


# In[ ]:


X_2u['distance'] = np.where((droprepeatmasker['distance']>-200)&(droprepeatmasker['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='distance')
plt.title('Upstream tsne:5 with RepeatMasker filter by distance>-200')
plt.savefig("upstream_TSNE:5_RepeatMakser>-200.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d['distance'] = np.where((droprepeatmasker['distance']<2000)&(droprepeatmasker['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='distance')
plt.title('Downstream tsne:5 with RepeatMasker filter by distance<2000')
plt.savefig("downstream_TSNE:5_RepeatMakser<2000.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d['distance'] = np.where((droprepeatmasker['distance']<200)&(droprepeatmasker['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='distance')
plt.title('Downstream tsne:5 with RepeatMasker filter by distance<200')
plt.savefig("downstream_TSNE:5_RepeatMakser<200.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d1['distance'] = np.where((droprepeatmasker['distance']<2000)&(droprepeatmasker['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d1, x="a", y="b", hue='distance')
plt.title('Downstream tsne:50 with RepeatMasker filter by distance<2000')
plt.savefig("Downstream_TSNE:50_RepeatMakser<2000.pdf",dpi=299)
plt.show()


# In[ ]:


principalupstreamwinnum['distance'] = np.where((droprepeatmasker['distance']>-200)&(droprepeatmasker['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance')
plt.title('Upstream pca with Repeat Masker Distance >-200')
plt.savefig("Upstream_pca_RepeatMakser>-200.pdf",dpi=299)
plt.show()


# In[ ]:


principaldownstreamwinnum['distance'] = np.where((droprepeatmasker['distance']<200)&(droprepeatmasker['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance')
plt.title('Downstream pca with Repeat Masker Distance <200')
plt.savefig("Downstream_pca_RepeatMakser<200.pdf",dpi=299)
plt.show()


# In[ ]:


principaldownstreamwinnum['distance'] = np.where((droprepeatmasker['distance']<200)&(droprepeatmasker['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance')
plt.title('Downstream pca with Repeat Masker Distance')
plt.show()


# # assign to Classes such as LINE, Alu, HERV, MER, LTR, L2 (Repeat Masker)

# In[ ]:


principalupstreamwinnum['class'] =droprepeatmasker['class']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='class')
plt.title("upstream pca with repeatmasker")
plt.show()


# In[ ]:


# pca downstream 
principalupstreamwinnum['class'] =droprepeatmasker['class']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='class')
plt.title("upstream pca with repeatmasker")
plt.show()

# pca upstream
principaldownstreamwinnum['class'] =droprepeatmasker['class']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='class')
plt.title("downstream pca with Repeatmasker")
plt.show()

#tsne down
X_2d['class']=np.where(droprepeatmasker['class'].str.contains("LTR"),"LTR",droprepeatmasker['class'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='class')
plt.title('Downstream TSNE with repeatmasker')
plt.show()

#tsne up
X_2u['class']=np.where(droprepeatmasker['class'].str.contains("LTR"),"LTR","MER")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='class')
plt.title('Upstream TSNE with repeatmasker')
plt.show()

X_2u1['class']=np.where(droprepeatmasker['class'].str.contains("LTR"),"LTR","MER")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u1, x="a", y="b", hue='class')
plt.title('Upstream TSNE with repeatmasker')
plt.show()

X_2u2['class']=np.where(droprepeatmasker['class'].str.contains("LTR"),"LTR","MER")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u2, x="a", y="b", hue='class')
plt.title('Upstream TSNE with repeatmasker')
plt.show()


# In[ ]:


X_2d['class']=np.where(droprepeatmasker['class'].str.contains("LTR"),"LTR",droprepeatmasker['class'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='class')
plt.title('Downstream TSNE with repeatmasker')
plt.show()


# In[ ]:


X_2u['class']=np.where(droprepeatmasker['class'].str.contains("LTR"),"LTR","MER")
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='class')
plt.title('Upstream TSNE with repeatmasker')
plt.show()


# In[ ]:


droprepeatmasker


# # RepeatMakser LINE TSNE and PCA

# In[ ]:


X_2u['class'] = np.where(droprepeatmasker['class'].str.contains('LINE'),'LINE', droprepeatmasker['class'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='class')
plt.title('Upstream TSNE with repearmasker class LINE')
plt.show()


# In[ ]:


X_2d['class'] = np.where(droprepeatmasker['class'].str.contains('LINE'),'LINE', droprepeatmasker['class'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='class')
plt.title('Downstream TSNE with repearmasker class LINE')
plt.show()


# # RepeatMakser Alu TSNE and PCA

# In[ ]:


# downstream
X_2d['class'] = np.where(droprepeatmasker['class'].str.contains('Alu'),'Alu', droprepeatmasker['class'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='class')
plt.title('Downstream TSNE with repearmasker class Alu')
plt.show()


# In[ ]:


# upstream
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('LINE'),'LINE', droprepeatmasker['superclass'])
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('Alu'),'Alu', droprepeatmasker['superclass'])
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('LTR'),'LTR', droprepeatmasker['superclass'])
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('MER'),'MER', droprepeatmasker['superclass'])
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('HERV'),'HERV', droprepeatmasker['superclass'])
X_2u['superclass'] = droprepeatmasker['superclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='superclass')
plt.title('Upstream TSNE with repearmasker class Alu')
plt.show()


# In[ ]:


# downstream
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('LINE'),'LINE', droprepeatmasker['superclass'])
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('Alu'),'Alu', droprepeatmasker['superclass'])
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('LTR'),'LTR', droprepeatmasker['superclass'])
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('MER'),'MER', droprepeatmasker['superclass'])
droprepeatmasker['superclass'] = np.where(droprepeatmasker['class'].str.contains('HERV'),'HERV', droprepeatmasker['superclass'])
X_2d['superclass'] = droprepeatmasker['superclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='superclass')
plt.title('Downstream TSNE with repearmasker class Alu')
plt.show()


# # RepeatMakser SubAlu Class PCA and TSNE

# In[ ]:


droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluY'),'AluY', droprepeatmasker['subAluclass'])
droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluS'),'AluS', droprepeatmasker['subAluclass'])
droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluJ'),'AluJ', droprepeatmasker['subAluclass'])

X_2d['subAluclass'] = droprepeatmasker['subAluclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='subAluclass')
plt.title('Downstream TSNE with repearmasker subclass Alu')
plt.show()


# In[ ]:


droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluY'),'AluY', droprepeatmasker['subAluclass'])
droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluS'),'AluS', droprepeatmasker['subAluclass'])
droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluJ'),'AluJ', droprepeatmasker['subAluclass'])

X_2u['subAluclass'] = droprepeatmasker['subAluclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='subAluclass')
plt.title('Upstream TSNE with repearmasker subclass Alu')
plt.show()


# In[ ]:


droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluY'),'AluY', droprepeatmasker['subAluclass'])
droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluS'),'AluS', droprepeatmasker['subAluclass'])
droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluJ'),'AluJ', droprepeatmasker['subAluclass'])

principalupstreamwinnum['subAluclass'] = droprepeatmasker['subAluclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='subAluclass')
plt.title('Upstream PCA with repearmasker subclass Alu')
plt.show()


# In[ ]:


droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluY'),'AluY', droprepeatmasker['subAluclass'])
droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluS'),'AluS', droprepeatmasker['subAluclass'])
droprepeatmasker['subAluclass'] = np.where(droprepeatmasker['class'].str.contains('AluJ'),'AluJ', droprepeatmasker['subAluclass'])

principaldownstreamwinnum['subAluclass'] = droprepeatmasker['subAluclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='subAluclass')
plt.title('Downstream PCA with repearmasker subclass Alu')
plt.show()


# In[ ]:


droprepeatmasker['superclass']= None
droprepeatmasker


# In[ ]:


droprepeatmasker['subAluclass']= None
droprepeatmasker


# In[ ]:


# AluS, AluJ, AluY 
# flatten all the windows - instead of concat, sum/ mean  (cluster)
# 


# # chrXch38 Validation Coloring

# In[ ]:


import numpy as np
import pandas as pd
ch38 = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/chrXCh38/closestsortedchrXGRCh38.bed", sep="\t", header=None, names = ['chrX','1','2','chrX1','3','4','class','distance'])
ch38


# In[ ]:


ch38drop=ch38.drop_duplicates(subset=["1","2"],keep="first") 
ch38drop


# In[ ]:


# pca downstream 
principaldownstreamwinnum['class'] =ch38drop['class']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='class')
plt.title("downstream pca with ch38")
plt.show()

# pca upstream
principalupstreamwinnum['class'] =ch38drop['class']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='class')
plt.title("upstream pca with ch38")
plt.show()

#tsne down
X_2d['class']=np.where(ch38drop['class'].str.contains("P"),"P",ch38drop['class'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='class')
plt.title('Downstream TSNE with ch38')
plt.show()

#tsne up
X_2u['class']=np.where(ch38drop['class'].str.contains("P"),"P",ch38drop['class'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='class')
plt.title('Upstream TSNE with ch38')
plt.show()


# In[ ]:


ch38drop['superclass']= None
ch38drop


# In[ ]:


# upstream
ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('pELS'),'pELS', ch38drop['superclass'])
ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('CTCF-only'),'CTCF', ch38drop['superclass'])
ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('P'),'PLS', ch38drop['superclass'])
ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('DNase'),'DNase', ch38drop['superclass'])
X_2u['superclass'] = ch38drop['superclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='superclass')
plt.title('Upstream TSNE with ch38drop class Alu')
plt.show()


# In[ ]:


ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('pELS'),'pELS', ch38drop['superclass'])
ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('CTCF-only'),'CTCF', ch38drop['superclass'])
ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('P'),'PLS', ch38drop['superclass'])
ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('DNase'),'DNase', ch38drop['superclass'])
X_2d['superclass'] = ch38drop['superclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='superclass')
plt.title('Downstream TSNE with ch38drop class Alu')
plt.show()


# In[ ]:


ch38drop['superclass'] = np.where(ch38drop['class'].str.contains('CTCF-bound'),'CTCFbound', ch38drop['superclass'])
X_2u1['superclass'] = ch38drop['superclass']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u1, x="a", y="b", hue='superclass')
plt.title('Downstream TSNE with ch38drop class CTCFbound')
plt.show()


# In[ ]:


ch38drop


# In[ ]:


ch38drop.head(40)


# # ch38 CTCF BOUND Coloring 

# In[ ]:


ch38drop['CTCFbound']= None
ch38drop


# In[ ]:


#Upsteam tsne
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('DNase-H3K4me3,CTCF-bound'),'DNase-H3K4me3', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('PLS,CTCF-bound'),'PLS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('pELS,CTCF-bound'),'pELS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('dELS,CTCF-bound'),'dELS', ch38drop['CTCFbound'])
X_2u['CTCFbound'] = ch38drop['CTCFbound']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='CTCFbound')
plt.title('Downstream TSNE with ch38drop class CTCFbound')
plt.show()


# In[ ]:


#Downsteam tsne
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('DNase-H3K4me3,CTCF-bound'),'DNase-H3K4me3', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('PLS,CTCF-bound'),'PLS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('pELS,CTCF-bound'),'pELS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('dELS,CTCF-bound'),'dELS', ch38drop['CTCFbound'])
X_2d['CTCFbound'] = ch38drop['CTCFbound']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='CTCFbound')
plt.title('Downstream TSNE with ch38drop class CTCFbound')
plt.show()


# In[ ]:


#downsteam pca
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('DNase-H3K4me3,CTCF-bound'),'DNase-H3K4me3', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('PLS,CTCF-bound'),'PLS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('pELS,CTCF-bound'),'pELS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('dELS,CTCF-bound'),'dELS', ch38drop['CTCFbound'])
principaldownstreamwinnum['CTCFbound'] = ch38drop['CTCFbound']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='CTCFbound')
plt.title('Downstream principaldownstreamwinnum with ch38drop class CTCFbound')
plt.show()


# In[ ]:


#upsteam pca
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('DNase-H3K4me3,CTCF-bound'),'DNase-H3K4me3', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('PLS,CTCF-bound'),'PLS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('pELS,CTCF-bound'),'pELS', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('CTCF-only,CTCF-bound'),'CTCFonly', ch38drop['CTCFbound'])
ch38drop['CTCFbound'] = np.where(ch38drop['class'].str.contains('dELS,CTCF-bound'),'dELS', ch38drop['CTCFbound'])
principalupstreamwinnum['CTCFbound'] = ch38drop['CTCFbound']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='CTCFbound')
plt.title('Downstream principalupstreamwinnum with ch38drop class CTCFbound')
plt.show()


# # filter by the distance (ch38)

# In[ ]:


X_2u['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='distance')
plt.title( 'Upstream TSNE with ch38drop distance of larger than -2000')
plt.savefig("Upstream TSNE withvalidationdatadistance>-2000.pdf",dpi=299)
plt.show()


# In[ ]:


X_2u1['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u1, x="a", y="b", hue='distance')
plt.title('Upstream TSNE:50 with ch38drop distance of larger than -2000')
plt.savefig("Upstream TSNE:50 withvalidationdatadistance>-2000.pdf",dpi=299)
plt.show()


# In[ ]:





# In[ ]:


principalupstreamwinnum['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance')
plt.title('Upstream pca with ch38drop distance of larger than -2000')
plt.savefig("Upstreampcawithvalidationdatadistance>-2000.pdf",dpi=299)
plt.show() 


# In[ ]:


principalupstreamwinnum['distance'] = np.where((ch38drop['distance']>-1000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance')
plt.title('Upstream pca with ch38drop distance of larger than -1000')
plt.savefig("Upstreampcawithvalidationdatadistance>-1000.pdf",dpi=299)
plt.show()


# In[ ]:


principalupstreamwinnum['distance'] = np.where((ch38drop['distance']>-500)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance')
plt.title('Upstream pca with ch38drop distance of larger than -500')
plt.savefig("Upstream pca withvalidationdatadistance>-500.pdf",dpi=299)
plt.show()


# # validation Downstream (filter by the distance)

# In[ ]:


# 2000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<2000)&(ch38drop['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance')
plt.title('Downstream pca with ch38drop distance of smaller than 2000')
plt.savefig("Downstreampcawithvalidationdatadistance<000.pdf",dpi=299)
plt.show()


# In[ ]:


# 50000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<50000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance')
plt.title('Downstream pca with ch38drop distance of smaller than 50000')
plt.savefig("Downstreampcawithvalidationdatadistance<50000.pdf",dpi=299)
plt.show()


# In[ ]:


# 5000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<5000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance')
plt.title('Downstream pca with ch38drop distance of smaller than 5000')
plt.savefig("Downstreampcawithvalidationdatadistance<5000.pdf",dpi=299)
plt.show()


# In[ ]:


# 1000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<1000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance')
plt.title('Downstream pca with ch38drop distance of smaller than 1000')
plt.savefig("Downstreampcawithvalidationdatadistance<1000.pdf",dpi=299)
plt.show()


# In[ ]:


# 500
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<500)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance')
plt.title('Downstream pca with ch38drop distance of smaller than 500')
plt.savefig("Downstreampcawithvalidationdatadistance<500.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d['distance'] = np.where((ch38drop['distance']<2000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='distance')
plt.title('Downstream TSNE with ch38drop distance of smaller than 2000')
plt.savefig("DownstreamTSNEwithvalidationdatadistance<2000.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d['distance'] = np.where((ch38drop['distance']<1000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='distance')
plt.title('Downstream TSNE with ch38drop distance of smaller than 1000')
plt.savefig("DownstreamTSNEwithvalidationdatadistance<1000.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d['distance'] = np.where((ch38drop['distance']<500)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='distance')
plt.title('Downstream TSNE with ch38drop distance of smaller than 500')
plt.savefig("DownstreamTSNEwithvalidationdatadistance<500.pdf",dpi=299)
plt.show()


# # UMAP Downstream

# In[60]:


import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[61]:


get_ipython().system('conda install -c conda-forge umap-learn -y')


# In[62]:


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
# In[63]:


get_ipython().system('pip install umap-learn')


# In[64]:


Downstreamdropwin


# In[65]:


Upstreamdropwin


# In[66]:


Upstreamdropwin.to_csv("Upstreamdropwin.tsv", index= None)
Downstreamdropwin.to_csv("Downstreamdropwin.tsv", index= None)


# In[67]:


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


# In[68]:


Xdowndf=pd.DataFrame(Xdown)


# In[69]:


Xdowndf


# In[70]:


plt.scatter(x=Xdowndf[0],y=Xdowndf[1])
plt.show()


# In[71]:



## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=Xdowndf,x=Xdowndf[0],y=Xdowndf[1], hue=Xdowndf[0])


# In[72]:


for key in translater.keys():

    Xdowndf[key] = sumTR[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdowndf, x=Xdowndf[0],y=Xdowndf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream 5kb-prereduction")
    plt.savefig("seqClass" + key+ "DOwnstream_umap(prereduct).pdf",dpi=299)
    plt.show()
    


# In[73]:


Xdowndf.astype(float)


# In[74]:


Xdowndf 


# In[75]:


plt.hist(Xdowndf.astype(float))


# In[76]:


from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


# In[77]:


for key in translater.keys():

    Xdowndf[key] = sumTR[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdowndf, x=Xdowndf[0],y=Xdowndf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream 5kb-prereduction")
    plt.savefig("seqClass" + key+ "Downstream_umap(prereduct).pdf",dpi=299)
    plt.show()


# In[78]:


Xdowndf


# In[79]:


umap2dimesiondown=pd.DataFrame(Xdown)


# In[80]:


umap2dimesiondown


# # UMAP upstream

# In[81]:


Downstreamdropwin
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


# In[82]:


XUpdf=pd.DataFrame(XUp)


# In[83]:


for key in translater.keys():

    XUpdf[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=XUpdf, x=XUpdf[0],y=XUpdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Upstream 5kb-prereduction")
    plt.savefig("seqClass" + key+ "Upstream_umap(prereduct).pdf",dpi=299)
    plt.show()


# # Plt Sum of SEQ TR DownStream

# In[85]:


umap2dimsionup=pd.DataFrame(XUp)


# In[86]:


dfUpwinnum


# In[87]:


dfDownwinnum


# In[88]:


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


# In[89]:


for tr_id in range(9,69):
    plt.plot(sumTRpltdown[tr_id])
    plt.title(tr_id)
    plt.show()


# In[90]:


dfsumTRpltdown=pd.DataFrame(sumTRpltdown).T


# In[91]:


dfsumTRpltdown


# In[92]:



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


# In[93]:


Xdownumapdf=pd.DataFrame(Xdownumap)


# In[94]:


# 
for key in translater.keys():

    Xdownumapdf[key] = sumTR[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdownumapdf, x=Xdownumapdf[0],y=Xdownumapdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream 5kb")
    plt.savefig("seqClass" + key+ "downstream_umap.pdf",dpi=299)
    plt.show()


# In[95]:


reductdownumap2d=pd.DataFrame(Xdownumap)


# # Plt Sum of SEQ TR UpStream

# In[96]:


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
    


# In[97]:


sumTRpltup


# In[98]:


dfsumTRpltup=pd.DataFrame(sumTRpltup).T


# In[99]:



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


# In[100]:


dfsumTRpltup


# In[ ]:


dfsumTRpltup.to_csv("dfsumTRpltup", index= None)
dfsumTRpltdown.to_csv("dfsumTRpltdown", index= None)


# In[101]:


Xupumapdf=pd.DataFrame(Xupumap)


# In[102]:


# 
for key in translater.keys():

    Xupumapdf[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xupumapdf, x=Xupumapdf[0],y=Xupumapdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Upstream 5kb")
    plt.savefig("seqClass" + key+ "Upstream_umap.pdf",dpi=299)
    plt.show()
    


# In[103]:


reductupumap2d=pd.DataFrame(Xupumap)


# In[ ]:


reductupumap2d
reductdownumap2d


# # Validation Data UMAP (Downstream)

# In[104]:


reductdownumap2d['distance'] = np.where((ch38drop['distance']<500)&(ch38drop['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d[0],y=reductdownumap2d[1], hue='distance')
plt.title("umap_Downstream 5kb-validationdata distance smaller than 500")
plt.savefig("umap_Downstream 5kb-validation-dist < 500.pdf",dpi=299)
plt.show()


# In[ ]:


reductdownumap2d['distance'] = np.where((ch38drop['distance']<5000)&(ch38drop['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d[0],y=reductdownumap2d[1], hue='distance')
plt.title("umap_Downstream 5kb-validationdata distance smaller than 5000")
plt.savefig("umap_Downstream 5kb-validation-dist < 5000.pdf",dpi=299)
plt.show()


# In[ ]:


reductdownumap2d['distance'] = np.where((ch38drop['distance']<100)&(ch38drop['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d[0],y=reductdownumap2d[1], hue='distance')
plt.title("umap_Downstream 5kb-validationdata distance smaller than 100")
plt.savefig("umap_Downstream 5kb-validation-dist < 100.pdf",dpi=299)
plt.show()


# In[ ]:


reductdownumap2d['distance'] = np.where((ch38drop['distance']<1000)&(ch38drop['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d[0],y=reductdownumap2d[1], hue='distance')
plt.title("umap_Downstream 5kb-validationdata distance smaller than 1000")
plt.savefig("umap_Downstream 5kb-validation-dist < 1000.pdf",dpi=299)
plt.show()


# In[ ]:


reductdownumap2d['distance'] = np.where((ch38drop['distance']<2000)&(ch38drop['distance']>0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d[0],y=reductdownumap2d[1], hue='distance')
plt.title("umap_Downstream 5kb-validationdata distance smaller than 2000")
plt.savefig("umap_Downstream 5kb-validation-dist < 2000.pdf",dpi=299)
plt.show()


# # Validation Data UMAP (Upstream)

# In[ ]:


reductupumap2d['distance'] = np.where((ch38drop['distance']>-500)&(ch38drop['distance']<0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d, x=reductupumap2d[0],y=reductupumap2d[1], hue='distance')
plt.title('Upstream UMAP with ch38drop larger than -500')
plt.show()


# In[ ]:


reductupumap2d['distance'] = np.where((ch38drop['distance']>-100)&(ch38drop['distance']<0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d, x=reductupumap2d[0],y=reductupumap2d[1], hue='distance')
plt.title("umap_Upstream 5kb-validationdata distance larger than -100")
plt.savefig("umap_Upstream 5kb-validation-dist >-100.pdf",dpi=299)
plt.show()


# In[ ]:


reductupumap2d['distance'] = np.where((ch38drop['distance']>-5000)&(ch38drop['distance']<0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d, x=reductupumap2d[0],y=reductupumap2d[1], hue='distance')
plt.title("umap_Upstream 5kb-validationdata distance larger than -5000")
plt.savefig("umap_Upstream 5kb-validation-dist >-5000.pdf",dpi=299)
plt.show()


# In[ ]:


reductupumap2d['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d, x=reductupumap2d[0],y=reductupumap2d[1], hue='distance')
plt.title("umap_Upstream 5kb-validationdata distance larger than -2000")
plt.savefig("umap_Upstream 5kb-validation-dist >-2000.pdf",dpi=299)
plt.show()


# In[ ]:


reductupumap2d['distance'] = np.where((ch38drop['distance']>-1000)&(ch38drop['distance']<0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d, x=reductupumap2d[0],y=reductupumap2d[1], hue='distance')
plt.title("umap_Upstream 5kb-validationdata distance larger than -1000")
plt.savefig("umap_Upstream 5kb-validation-dist >-1000.pdf",dpi=299)
plt.show()


#  tsne umap and pca are dimensionality reduction techniques, so they are used to project high dimensional data to lower dimensions
# which is helpful for visualization
# 
# clustering algorithms computationally group data points by their similarity, so the dimensionality reduced data would be an input to a clustering algorithm
# the reason we want to use a clustering algorithm (or community detection, they are kinda used interchangeably) is to identfy groups in our data that we may be able to see visually

# In[106]:


#umap2dimsiondown.to_csv("umap2dimsiondown.tsv", index= None)
umap2dimsionup.to_csv("umap2dimensiondup.tsv", index= None)
reductupumap2d.to_csv("reductupumap2d.tsv", index= None)
reductdownumap2d.to_csv("reductdownumap2d.tsv", index= None)


# # Kmean 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# # Kmean Upstream Matirx

# In[ ]:


Upstreamdropwinscale=StandardScaler().fit_transform(Upstreamdropwin)


# In[ ]:


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


# In[ ]:


#instantiate the k-means class, using optimal number of clusters
kmeans = KMeans(init="random", n_clusters=3, n_init=10, random_state=1)

#fit k-means algorithm to data
kmeans.fit(Upstreamdropwinscale)

#view cluster assignments for each observation
kmeans.labels_


# In[ ]:


upscaledff=pd.DataFrame(Upstreamdropwinscale)


# In[ ]:


upscaledff


# In[ ]:


#append cluster assingments to original DataFrame
upscaledff['cluster'] = kmeans.labels_

#view updated DataFrame
print(upscaledff)


# In[ ]:


upscaledff['cluster'].unique()


# # Kmean Matrix coloring 

# In[ ]:


for key in translater.keys():

    upscaledff[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=upscaledff, x="a", y="b", hue=key)
    plt.title(key)
    plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=upscaledff, x=upscaledff[0], y=upscaledff[1], hue='cluster')
plt.title("kmeans, Upstream 5kb")
plt.savefig("kmeans,Upstream_fullmatrix.pdf",dpi=299)
plt.show()


# # Kmean Upstream PCA

# In[ ]:


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


# In[ ]:


principalupstreamwinnumscaledf=pd.DataFrame(principalupstreamwinnumscale)


# In[ ]:


#append cluster assingments to original DataFrame\
principalupstreamwinnumscaledf=pd.DataFrame(principalupstreamwinnumscale)
principalupstreamwinnumscaledf['cluster'] = kmeans.labels_

#view updated DataFrame
print(principalupstreamwinnumscaledf)


# # Kmean PCA coloring Upstream

# In[ ]:


for key in translater.keys():

    principalupstreamwinnumscaledf[key] = sumTRUP[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnumscaledf, x=principalupstreamwinnumscaledf[0], y=principalupstreamwinnumscaledf[1], hue=key)
    plt.title(key)
    plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnumscaledf, x=principalupstreamwinnumscaledf[0], y=principalupstreamwinnumscaledf[1], hue='cluster')
plt.title("kmeans, Upstream pca 5kb")
plt.savefig("kmeans,Upstream_pca.pdf",dpi=299)
plt.show()


# In[ ]:


## Save as TSV // read as CSV


# In[ ]:


#principalupstreamwinnumscaledf.to_csv("principalupstreamwinnumscaledf", index= None)
principalupstreamwinnumscaledf=pd.read_csv('principalupstreamwinnumscaledf')
#upscaledff.to_csv("upscaledff", index= None)
upscaledff=pd.read_csv('upscaledff')


# In[ ]:





# # Kmean Upstream TSNE

# In[ ]:


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


# In[ ]:


tsneupscaledf


# In[ ]:


#append cluster assingments to original DataFrame\
tsneupscaledf=pd.DataFrame(tsneupscaledf)
tsneupscaledf['cluster'] = kmeans.labels_

#view updated DataFrame
print(tsneupscaledf)


# In[ ]:


tsneupscaledf['cluster'] = kmeans.labels_


# In[ ]:


tsneupscaledf


# In[ ]:


tsneupscaledf['cluster'].unique()


# In[ ]:


tsneupscaledf['cluster'].value_counts()


# # Kmean TSNE coloring Upstream 

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsneupscaledf, x=tsneupscaledf[0], y=tsneupscaledf[1], hue='cluster')
plt.title("kmeans, Upstream 5kb tsne")
plt.savefig("kmeans,Upstream_tsne.pdf",dpi=299)
plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsneupscaledf[tsneupscaledf['cluster']== 2], x=tsneupscaledf[tsneupscaledf['cluster']== 2][0], y=tsneupscaledf[tsneupscaledf['cluster']== 2][1], hue='cluster')
plt.title("tsne, kmean")
plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsneupscaledf[tsneupscaledf['cluster']== 8], x=tsneupscaledf[tsneupscaledf['cluster']== 8][0], y=tsneupscaledf[tsneupscaledf['cluster']== 8][1], hue='cluster')
plt.title("tsne, kmean")
plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsneupscaledf[tsneupscaledf['cluster']== 0], x=tsneupscaledf[tsneupscaledf['cluster']== 0][0], y=tsneupscaledf[tsneupscaledf['cluster']== 0][1], hue='cluster')
plt.title("tsne, kmean")
plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsneupscaledf[tsneupscaledf['cluster']== 9], x=tsneupscaledf[tsneupscaledf['cluster']== 9][0], y=tsneupscaledf[tsneupscaledf['cluster']== 9][1], hue='cluster')
plt.title("tsne, kmean")
plt.show()


# In[ ]:


umap2dimensiondup=pd.read_csv("umap2dimensiondup.tsv", header=None, names=['0','1'])


# In[ ]:


umap2dimensiondup


# In[ ]:


umap2dimensiondup.drop(index=umap2dimensiondup.index[0], axis=0, inplace=True)
# on UMAP


# # UMAP Upstream

# In[ ]:


# on UMAP
dfsumTRpltupscale=StandardScaler().fit_transform(umap2dimensiondup)
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


# In[ ]:


dfsumTRpltupscalef=pd.DataFrame(dfsumTRpltupscale)


# In[ ]:



dfsumTRpltupscalef=pd.DataFrame(dfsumTRpltupscale)
dfsumTRpltupscalef['cluster'] = kmeans.labels_

#view updated DataFrame
print(dfsumTRpltupscalef)


# # Kmean UMAP coloring Upstream

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=dfsumTRpltupscalef, x=dfsumTRpltupscalef[0], y=dfsumTRpltupscalef[1], hue='cluster')
plt.title("kmeans, Upstream 5kb umap")
plt.savefig("kmeans,Upstream_umap.pdf",dpi=299)
plt.show()


# In[ ]:


dfsumTRpltupscalef['cluster'] = ch38drop['cluster']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=dfsumTRpltupscalef, x="a", y="b", hue='cluster')
plt.title("UMAP with ch38, kmean")
plt.show()


# # Downstream Kmeans

# In[ ]:


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


# In[ ]:


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
plt.title("kmean, fullmatrix_Downstream 5kb")
plt.savefig("downstream_kmean_fullmatrix.pdf",dpi=299)
plt.show()


# # Downstream PCA kmeans 

# In[ ]:


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


# In[ ]:


#append cluster assingments to original DataFrame\
principaldownstreamwinnumscaledf=pd.DataFrame(principaldownstreamwinnumscale)
principaldownstreamwinnumscaledf['cluster'] = kmeans.labels_

#view updated DataFrame
print(principaldownstreamwinnumscaledf)


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnumscaledf, x=principaldownstreamwinnumscaledf[0], y=principaldownstreamwinnumscaledf[1], hue='cluster')
plt.title("kmean, pca_Downstream 5kb")
plt.savefig("downstream_kmean_pca.pdf",dpi=299)
plt.show()


# # DownStream Tsne Kmeans

# In[ ]:


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


# In[ ]:


tsnedownscaledf=pd.DataFrame(tsnedownscaledf)


# In[ ]:


tsnedownscaledf['cluster'] = kmeans.labels_


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=tsnedownscaledf, x=tsnedownscaledf[0], y=tsnedownscaledf[1], hue='cluster')
plt.title("kmean, tsne_Downstream 5kb")
plt.savefig("downstream_kmean_tsne.pdf",dpi=299)
plt.show()


# In[ ]:


X_2d


# In[ ]:


umap2dimsiondown=pd.read_csv("umap2dimsiondown.tsv", header=None, names=['1','2'])
umap2dimsiondown.drop(index=umap2dimsiondown.index[0], axis=0, inplace=True)


# In[ ]:


umap2dimsiondown


# In[ ]:


tsnedownscaledf


# In[ ]:


dfsumTRpltup


# In[ ]:


dfsumTRpltdown


# In[ ]:


Xdownumapdf


# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=tsnedownscaledf, x=tsnedownscaledf[0], y=tsnedownscaledf[1], hue='cluster')
# plt.title("tsne, kmean")
# plt.show()
# 
# 

# In[ ]:


# on UMAP
Xdownumapdfscale=StandardScaler().fit_transform(umap2dimsiondown)
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


# In[ ]:


Xdownumapdfscale=pd.DataFrame(Xdownumapdfscale)
Xdownumapdfscale


# In[ ]:


Xdownumapdfscale['cluster'] = kmeans.labels_


# In[ ]:


Xdownumapdfscale


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data= Xdownumapdfscale, x= Xdownumapdfscale[0], y= Xdownumapdfscale[1], hue='cluster')
plt.title("kmean, umap_Downstream 5kb")
plt.savefig("downstream_kmean_umap.pdf",dpi=299)
plt.show()


# In[ ]:


Upstreamdropwin


# # Louvain Community detection Algorithmn

# In[ ]:


get_ipython().run_line_magic('pip', 'install scanpy -q')
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


adata = sc.read('updropwin.tsv')

 # extract the UMAP coordinates for each cell
kmeans = KMeans(n_clusters=4, random_state=0).fit(Upstreamdropwinscale) # fix the random state for reproducibility

adata.obs['kmeans'] = kmeans.labels_ # retrieve the labels and add them as a metadata column in our AnnData object
adata.obs['kmeans'] = adata.obs['kmeans'].astype(str)

#sc.pl.umap(adata, color='kmeans') 


# In[ ]:


sc.tl.louvain(adata)


# In[ ]:


Downstreamdropwin.to_csv('downdropwin.tsv', sep="\t")


# In[ ]:


Upstreamdropwin.to_csv('updropwin.tsv', sep="\t")


# In[ ]:


sc.tl.louvain(adata)
sc.pl.umap(adata, color='louvain')


# In[ ]:


edge = [(1,2),(1,3),(1,4),(1,5),(1,6),(2,7),(2,8),(2,9)]
G = nx.Graph()
G.add_edges_from(edge)
# retrun partition as a dict
partition = community_louvain.best_partition(G)
# visualization
pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=100,cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()


# # Louvain Full Matrix (garbage)

# In[ ]:


G = nx.Graph()
G.add_edges_from(Upstreamdropwin)
# retrun partition as a dict
partition = community_louvain.best_partition(G)
# visualization
pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=100,cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()


# # high_noise_clustering (Louvain)
# 

# In[107]:


get_ipython().system(' pip install community')


# In[108]:


get_ipython().system(' pip install igraph')


# In[109]:


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

plt.ion()
plt.show()


# In[110]:


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

plt.ion()
plt.show()


# In[111]:


dfDownwinnum


# In[112]:


dfUpwinnum


# In[113]:


Downstreamdropwin


# In[114]:


Upstreamdropwin


# In[115]:


n_clusters = 6
n_features=3050
n_samples=14500
random_state = 42


updata, truth = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state, n_features=n_features)
updata = preprocessing.MinMaxScaler().fit_transform(Upstreamdropwin)

# Plot
plt.scatter(updata[:, 0], updata[:, 1], s=50, c = truth, cmap = 'viridis')
plt.title(f"Example of a mixture of {n_clusters} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2");


# In[116]:


Downstreamdropwin


# In[117]:


downdata


# In[ ]:


pd.read_csv('downdropwin.tsv',sep="\t", header=None)


# In[121]:


n_clusters = 6
n_features=3050
n_samples=14500
random_state = 42


downdata, truth = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state, n_features=n_features)
downdata = preprocessing.MinMaxScaler().fit_transform(Downstreamdropwin)
downdata 

# Plot
plt.scatter(downdata[:, 0], downdata[:, 1], s=50, c = truth, cmap = 'viridis')
plt.title(f"Example of a mixture of {n_clusters} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2");


# In[122]:


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


# In[123]:


downpredictionbyw = cluster_by_connectivity(downdata, resolution_parameter = 0.5)
Counter(downpredictionbyw)


# In[124]:


downdata


# In[125]:


downdata=pd.DataFrame(downdata)


# In[150]:



cnt = 0
resolution_parameter = {'0.3': 0.3, '0.5': 0.5, '0.6': 0.6, '0.7': 0.7, '0.8': 0.8, '0.9': 0.9,'1': 1}
resolution_result ={}
for key in resolution_parameter.keys():
    resolution_result[key]  = cluster_by_connectivity(downdata, resolution_parameter = resolution_parameter[key])


# In[ ]:


for key in resolution_parameter.keys():
    plt.plot(resolution_result[key])
    plt.title(key)
    plt.savefig('louvain_downstream_resolutionparameter', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    
    plt.show()


# # cluster by the distanceMatrix-weighted (downstream)

# In[132]:


downdata=pd.DataFrame(downdata)
downdata


# In[ ]:


updata


# In[ ]:


downdistanceMatrix =  euclidean_distances(downdata, downdata)
print(downdistanceMatrix.shape)


# In[ ]:


downdistanceMatrix =  euclidean_distances(downdata, downdata)
print(downdistanceMatrix.shape)
updistanceMatrix =  euclidean_distances(updata, updata)
print(updistanceMatrix.shape)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


upprediction = cluster_by_distance_matrix(updistanceMatrix)
Counter(upprediction)


# In[ ]:


from scipy.spatial import distance_matrix


# In[ ]:


downprediction = cluster_by_distance_matrix(downdistanceMatrix)
Counter(downprediction)


# In[ ]:


adjusted_rand_score(truth, downprediction)


# In[ ]:


adjusted_rand_score(truth, downprediction)
adjusted_rand_score(truth, upprediction)


# distance based partitioning/ compute distance matrix:

# make predicition:

# In[ ]:


dfdownprediction=pd.DataFrame(downprediction)
dfdownprediction


# In[ ]:


dfupprediction=pd.DataFrame(upprediction)
dfupprediction


# In[ ]:


downresult = pd.concat([Downstreamdropwin, dfdownprediction], axis=1)


# In[ ]:


upresult = pd.concat([Upstreamdropwin, dfupprediction], axis=1)


# In[ ]:


upresult.columns = [*upresult.columns[:-1], 'up_prediction']


# In[ ]:


downresult.columns = [*downresult.columns[:-1], 'down_prediction']


# In[ ]:


downresult


# In[ ]:


upresult


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=upresult, x=upresult['0'], y=upresult['1'], hue='up_prediction')
plt.title("14500/3050matrix, louvain")
plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=downresult, x=downresult['0'], y=downresult['1'], hue='down_prediction')
plt.title("14500/3050matrix, louvain")
plt.show()


# constructing the list, using counter range of value >10

# In[ ]:


Counter(prediction)


# In[ ]:



#print(DownstreamMatrix.columns)
downstream_res = {}


exclude =[]

for key,value in Counter(downprediction).items():
    if (10> value):
        exclude.append(key)
        
principaldownstreamwinnum['louvainweightresult']=np.where(pd.Series(downprediction).isin(exclude),-1,downprediction)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='louvainweightresult')
plt.title("louvain downstream weighted on pca")
plt.savefig("louvain_downstream_weighted_pca.pdf",dpi=299)
plt.show()


# In[ ]:



#print(DownstreamMatrix.columns)
upstream_res = {}


exclude =[]

for key,value in Counter(upprediction).items():
    if (10> value):
        exclude.append(key)
        
principalupstreamwinnum['louvainweightresult']=np.where(pd.Series(upprediction).isin(exclude),-1,upprediction)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='louvainweightresult')
plt.title("louvain upstream weighted on pca")
plt.savefig("louvain_upstream_weighted_pca.pdf",dpi=299)
plt.show()


# In[ ]:


upstream_res = {}


exclude =[]

for key,value in Counter(upprediction).items():
    if (10> value):
        exclude.append(key)
        
X_2u['louvainweightresult']=np.where(pd.Series(upprediction).isin(exclude),-1,upprediction)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='louvainweightresult')
plt.title("louvain upstream weighted on tsne")
plt.savefig("louvain_upstream_weighted_tsne.pdf",dpi=299)
plt.show()


# In[ ]:


downstream_res = {}


exclude =[]

for key,value in Counter(downprediction).items():
    if (10> value):
        exclude.append(key)
        
X_2d['louvainweightresult']=np.where(pd.Series(downprediction).isin(exclude),-1,downprediction)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2d, x="a", y="b", hue='louvainweightresult')
plt.title("louvain downstream weighted on tsne")
plt.savefig("louvain_downstream_weighted_tsne.pdf",dpi=299)
plt.show()


# In[ ]:


downstream_res = {}


exclude =[]

for key,value in Counter(downprediction).items():
    if (10> value):
        exclude.append(key)
Xdownumapdfscale['louvainweightresult']=np.where(pd.Series(downprediction).isin(exclude),-1,downprediction)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=Xdownumapdfscale, x=Xdownumapdfscale[0], y=Xdownumapdfscale[1], hue='louvainweightresult')
plt.title("louvain downstream weighted on umap")
plt.savefig("louvain_downstream_weighted_umap.pdf",dpi=299)
plt.show()


# In[ ]:


upstream_res = {}


exclude =[]

for key,value in Counter(upprediction).items():
    if (10> value):
        exclude.append(key)
dfsumTRpltupscalef['louvainweightresult']=np.where(pd.Series(downprediction).isin(exclude),-1,uprediction)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=dfsumTRpltupscalef, x=dfsumTRpltupscalef[0], y=dfsumTRpltupscalef[1], hue='louvainweightresult')
plt.title("louvain upstream weighted on umap")
plt.savefig("louvain_upstream_weighted_umap.pdf",dpi=299)
plt.show()


# # cluster by connectivity (downstream) 

# In[ ]:


def cluster_by_connectivity(data, neighbors = 10, resolution_parameter = 1):
    """
    This method partitions input data by applying the louvain algorithm
    on the connectivity binary matrix returned by the kneighbors graph.
  

  """
    A = kneighbors_graph(data, neighbors, mode='connectivity', include_self=True)
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


# In[ ]:


increasedprediction = cluster_by_distance_matrix(distanceMatrix)
Counter(increasedprediction)


# In[ ]:


dfincreasedprediction=pd.DataFrame(increasedprediction)


# In[ ]:


result = pd.concat([Downstreamdropwin, dfincreasedprediction], axis=1)


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=result, x=result['0'], y=result['1'], hue='increasedprediction')
plt.title("14500/3050matrix, louvain")
plt.show()


# In[ ]:


result.columns = [*result.columns[:-1], 'increasedprediction']


# # Downstream connectivity plotting/ coloring (tsne, pca, and umap)

# In[ ]:


for key in resolution_parameter.keys():
    downdata[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=downdata, x=downdata[0], y=downdata[1], hue=key)
    plt.title(key)
    plt.show()


# In[ ]:


umap2dimensiondup


# In[ ]:


for key in resolution_parameter.keys():

    principaldownstreamwinnum[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue=key)
    plt.title("Louvain Clustering, resolution: "+ key + ", Downstream 5kb")
    plt.savefig("louvain_res_" + key+ "_pca_down.pdf",dpi=299)
    plt.show()


# In[ ]:


for key in resolution_parameter.keys():

    X_2d[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d, x="a", y="b", hue=key)
    plt.title("Louvain Clustering, resolution: "+ key + ", Downstream 5kb")
    plt.savefig("louvain_res_" + key+ "_tsne_down.pdf",dpi=299)
    
    plt.show()


# In[ ]:


for key in resolution_parameter.keys():

    Xdownumapdfscale[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdownumapdfscale, x=Xdownumapdfscale[0],y=Xdownumapdfscale[1], hue= resolution_result[key])
    plt.title("Louvain Clustering, resolution: "+ key + ", Downstream 5kb")
    plt.savefig("louvain_res_" + key+ "_umap_down.pdf",dpi=299)
    plt.show()


# In[ ]:


for key in resolution_parameter.keys():

    X_2u[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2u, x="a", y="b", hue=key)
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream 5kb")
    plt.savefig("louvain_res_" + key+ "_tsne_up.pdf",dpi=299)
    
    plt.show()


# In[ ]:





# # Upstream High dimension Louvain Analysis

# In[134]:


Upstreamdropwin


# In[135]:


n_clusters = 6
n_features=3050
n_samples=14500
random_state = 42


updata, truth = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state, n_features=n_features)
updata = preprocessing.MinMaxScaler().fit_transform(Upstreamdropwin)
updata 

# Plot
plt.scatter(updata[:, 0], updata[:, 1], s=50, c = truth, cmap = 'viridis')
plt.title(f"Example of a mixture of {n_clusters} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2");


# In[138]:


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


# In[142]:


def cluster_by_connectivity(data, neighbors = 10, resolution_parameter = 1):
    """
    This method partitions input data by applying the louvain algorithm
    on the connectivity binary matrix returned by the kneighbors graph.
  

  """
    A = kneighbors_graph(data, neighbors, mode='connectivity', include_self=True)
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


# In[139]:


uppredictionbyw = cluster_by_connectivity(updata, resolution_parameter = 0.5)
Counter(uppredictionbyw)


# In[140]:


updata=pd.DataFrame(updata)


# In[143]:



cnt = 0
resolution_parameter = {'0.3': 0.3, '0.5': 0.5, '0.6': 0.6, '0.7': 0.7, '0.8': 0.8, '0.9': 0.9,'1': 1}
resolution_result ={}
for key in resolution_parameter.keys():
    resolution_result[key]  = cluster_by_connectivity(updata, resolution_parameter = resolution_parameter[key])


# # Upstream connectivity plotting/ coloring (tsne, pca, and umap)

# In[145]:


principalupstreamwinnum


# In[ ]:


for key in resolution_parameter.keys():

    principalupstreamwinnum[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue=key)
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream 5kb")
    plt.savefig("louvain_res_" + key+ "_pca_up.pdf",dpi=299)
    plt.show()


# In[ ]:


for key in resolution_parameter.keys():

    dfsumTRpltupscalef[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=dfsumTRpltupscalef, x=dfsumTRpltupscalef[0],y=dfsumTRpltupscalef[1], hue= resolution_result[key])
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream 5kb")
    plt.savefig("louvain_res_" + key+ "_umap_up.pdf",dpi=299)
    plt.show()


# In[127]:


dfsumTRpltup.to_csv("dfsumTRpltup", index= None)
dfsumTRpltdown.to_csv("dfsumTRpltdown", index= None)


# In[141]:


reductdownumap2d
reductupumap2d


# In[ ]:


Xupumapdf


# In[ ]:



# on UMAP
reductdownumap=StandardScaler().fit_transform(reductdownumap2d)
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
    kmeans.fit(reductdownumap)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
dfreductdownumap=pd.DataFrame(reductdownumap)

dfreductdownumap=['cluster'] = kmeans.labels_

#view updated DataFrame
print(dfreductdownumap)


# In[ ]:



# on UMAP
reductupumap=StandardScaler().fit_transform(reductupumap2d)
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
    kmeans.fit(reductupumap)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
dfreductupumap=pd.DataFrame(reductupumap)

dfreductupumap['cluster'] = kmeans.labels_

#view updated DataFrame
print(dfreductupumap)


# In[149]:


for key in resolution_parameter.keys():

    reductupumap2d[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=reductupumap2d, x=reductupumap2d[0], y=reductupumap2d[1], hue=key)
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream 5kb")
    plt.savefig("louvain_res_" + key+ "_umap_sumTR_up.pdf",dpi=299)
    plt.show()


# In[151]:


for key in resolution_parameter.keys():

    reductdownumap2d[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d[0], y=reductdownumap2d[1], hue=key)
    plt.title("Louvain Clustering, resolution: "+ key + ", Downstream 5kb")
    plt.savefig("louvain_res_" + key+ "_umap_sumTR_down.pdf",dpi=299)
    plt.show()


# In[ ]:




