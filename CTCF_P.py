#!/usr/bin/env python
# coding: utf-8

# # chr9_30

# In[9]:


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

# tsne
X_2u2=  pd.read_csv("X_2u2_chr930.csv")
X_2u1= pd.read_csv("X_2u1_chr930.csv")
X_2u= pd.read_csv("X_2u_chr930.csv")

# pca
principalupstreamwinnum= pd.read_csv("principalupstreamwinnum_chr930")
principaldownstreamwinnum= pd.read_csv("principaldownstreamwinnum_chr930")
#full matrix
dfDownwinnum= pd.read_csv("dfDownwinnum_chr930")
dfUpwinnum= pd.read_csv("dfUpwinnum_chr930")
Upstreamdropwin = pd.read_csv("Upstreamdropwin_chr930.csv")
Downstreamdropwin = pd.read_csv("Downstreamdropwin_chr930")
#louvain
upresult= pd.read_csv("upresult_chr930.tsv")
downresult= pd.read_csv("downresult_chr930.tsv")
dfupprediction= pd.read_csv("dfupprediction_chr930.tsv")
dfdownprediction= pd.read_csv("dfdownprediction_chr930.tsv")
# save the dfs
dfsumTRpltup= pd.read_csv("dfsumTRpltup_chr930.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chr930.tsv")
umap2dimensionup_chr9= pd.read_csv("umap2dimensionup_chr930.tsv")
umap2dimensiondown_chr9= pd.read_csv("umap2dimensiondown_chr930.tsv")
# reduced 
reductdownumap2d= pd.read_csv("reductdownumap2d_chr930.tsv")
reductupumap2d_chr9= pd.read_csv("reductupumap2d_chr930.tsv")
# louvain byw
dfdownpredictionbyw= pd.read_csv("downpredictionbyw_chr930.tsv")
dfuppredictionbyw= pd.read_csv("uppredictionbyw_chr930.tsv")
# louvain result 
upresult= pd.read_csv("upresult_chr930.tsv")
downresult= pd.read_csv("downresult_chr930.tsv")


# In[10]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

mostimportant10= most_important10_differentmethod[:,0]


Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns

up_mostimportantsubset=Upstreamdropwin[mostimportant10]
up_mostimportantsubset

importantupresult = pd.concat([up_mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']

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


# In[11]:



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



# In[12]:



#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

up_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

up_mostimportant10= up_most_important10_differentmethod[:,0]
up_mostimportant10= list(up_mostimportant10)

mostimportantsubset=Upstreamdropwin[up_mostimportant10]
mostimportantsubset

importantupresult = pd.concat([mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']
 
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
                          
down_most_important10_differentmethod
Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 
down_mostimportantsubset=Downstreamdropwin[down_mostimportant10]
down_mostimportantsubset
importantdownresult = pd.concat([down_mostimportantsubset, dfdownprediction], axis=1)
importantdownresult.columns = [*importantdownresult .columns[:-1], 'p']

importantdownresult
importantupresult
meanupimportant = importantupresult.groupby('p').mean()
meanupdf=pd.DataFrame(meanupimportant)

print(meanupdf)
                          
                          
meandownimportant = importantdownresult.groupby('p').mean()
meandowndf=pd.DataFrame(meandownimportant)
meandowndf


countdown = importantdownresult.value_counts('p')
countup =importantupresult.value_counts('p')
dfcountup=pd.DataFrame(countup)
dfcountdown=pd.DataFrame(countdown)

dfcountupwhere = np.where((dfcountup[0]<10),False,True)
dfcountdownwhere = np.where((dfcountdown[0]<10),False,True)

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


# In[13]:


dfsumTRpltdown = np.log10(dfsumTRpltdown)
dfsumTRpltdown
dfsumTRpltup = np.log10(dfsumTRpltup)
dfsumTRpltup


# In[14]:


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
plt.savefig('upstream_mostimportant10_HeatMap_60seqclass_chr930.pdf', dpi=299)
sns.clustermap(upheatsum60)


# In[15]:


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
plt.savefig('downstream_mostimportant10_HeatMap_60seqclass_chr930.pdf', dpi=299)
sns.clustermap(downheatsum60)


# In[16]:


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
plt.title('upstream zscore_HeatMap_chr930')
plt.savefig('upstream_zscore__heatmap_chr930.pdf', dpi=299)
sns.heatmap(meanimportantzscoreup60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreup60)
reductupumap2d_chr9['p'] = dfuppredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='p',palette="tab10")
plt.show()


# In[17]:


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
plt.title('downstream zscore_HeatMap_chr930')
plt.savefig('downstream_zscore__heatmap_chr930.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60)
reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


# In[18]:


dfUpseqname = dfUpwinnum.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6','49':'losig1','50':'losig2','51':'losig3','52':'losig4'})
dfUpseqname
dfUpseqname= dfUpseqname.drop(columns=['TR_id', 'Win_num','Result'])
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
plt.title('upstream zscore_HeatMap_chr930')
plt.savefig('upstream_zscore__heatmap_chr930.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredup60seq)


# In[19]:


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
plt.title('downstream zscore_HeatMap_chr930')
plt.savefig('downstream_zscore__heatmap_chr930.pdf', dpi=299)
sns.heatmap(meanimportantzscoreddown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreddown60seq)


# In[20]:


reductupumap2d_chr9['p'] = dfuppredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='p',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


# In[21]:


dfsumTRpltup= pd.read_csv("dfsumTRpltup_chr930.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chr930.tsv")

dfUpseqname = dfsumTRpltup.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfUpseqname

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

reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()

reductupumap2d_chr9['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chr9['CTCF_high'] = np.where(reductupumap2d_chr9['CTCF']>10,True,False)
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


dfDownseqname = dfsumTRpltdown.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfDownseqname

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

dfUpseqname = dfsumTRpltup.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfUpseqname

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

reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>40,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>100,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()

reductupumap2d_chr9['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chr9['CTCF_high'] = np.where(reductupumap2d_chr9['CTCF']>40,True,False)
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>100,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


# In[22]:


plt.hist(reductdownumap2d['P'])


# In[23]:


plt.hist(reductdownumap2d['CTCF'])


# # chr9_30_5kb

# In[24]:


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

upresult= pd.read_csv("upresult_chr930_5kb.tsv")
downresult= pd.read_csv("downresult_chr930_5kb.tsv")


# In[25]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

mostimportant10= most_important10_differentmethod[:,0]


# In[26]:



Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns 


# In[27]:



Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns 

up_mostimportantsubset=Upstreamdropwin[mostimportant10]
up_mostimportantsubset


# In[28]:


importantupresult = pd.concat([up_mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']


# In[29]:



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


# In[30]:


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


# In[31]:



#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

up_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

up_mostimportant10= up_most_important10_differentmethod[:,0]
up_mostimportant10= list(up_mostimportant10)

mostimportantsubset=Upstreamdropwin[up_mostimportant10]
mostimportantsubset

importantupresult = pd.concat([mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']
 
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
                          
down_most_important10_differentmethod
Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 
down_mostimportantsubset=Downstreamdropwin[down_mostimportant10]
down_mostimportantsubset
importantdownresult = pd.concat([down_mostimportantsubset, dfdownprediction], axis=1)
importantdownresult.columns = [*importantdownresult .columns[:-1], 'p']

importantdownresult
importantupresult
meanupimportant = importantupresult.groupby('p').mean()
meanupdf=pd.DataFrame(meanupimportant)

print(meanupdf)
                          
                          
meandownimportant = importantdownresult.groupby('p').mean()
meandowndf=pd.DataFrame(meandownimportant)
meandowndf


countdown = importantdownresult.value_counts('p')
countup =importantupresult.value_counts('p')
dfcountup=pd.DataFrame(countup)
dfcountdown=pd.DataFrame(countdown)

dfcountupwhere = np.where((dfcountup[0]<10),False,True)
dfcountdownwhere = np.where((dfcountdown[0]<10),False,True)

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


# In[32]:


import pandas as pd

dfsumTRpltdown = np.log10(dfsumTRpltdown)
dfsumTRpltdown
dfsumTRpltup = np.log10(dfsumTRpltup)
dfsumTRpltup


# In[33]:


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


# In[34]:


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


# In[35]:


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
sns.clustermap(meanimportantzscoreup60)


# In[36]:



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
plt.title('downstream zscore_HeatMa_chr930_5kb')
plt.savefig('downstream_zscore__heatmap.pdf_chr930_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60)


# In[37]:


reductupumap2d_chr9['p'] = dfuppredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='p',palette="tab10")
plt.show()


# In[38]:




reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


# In[39]:


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


# In[40]:


dfUpseqname = dfUpwinnum.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6','49':'losig1','50':'losig2','51':'losig3','52':'losig4'})
dfUpseqname
dfUpseqname=dfUpseqname.drop(columns=['TR_id', 'Win_num','Result'])

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


# In[41]:


dfsumTRpltup= pd.read_csv("dfsumTRpltup_chr930_5kb.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chr930_5kb.tsv")


# In[42]:


dfUpseqname = dfsumTRpltup.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfUpseqname

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
sns.clustermap(meanimportantzscoredup60seq, square=False,figsize=(15.7,8.27), xticklabels=1)
sns.set(rc={'figure.figsize':(15.7,8.27)})


# In[43]:


dfDownseqname = dfsumTRpltdown.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfDownseqname

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


# In[44]:


reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()


# In[45]:


reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()


# In[46]:


reductupumap2d_chr9['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chr9['CTCF_high'] = np.where(reductupumap2d_chr9['CTCF']>10,True,False)
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='CTCF_high',palette="tab10")
plt.show()


# In[47]:


reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


# # chr9_3

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
import numpy as np
import pandas as pd
import numpy as np
X_2u2=  pd.read_csv("X_2u2_chr9.csv")
X_2u1= pd.read_csv("X_2u1_chr9.csv")
X_2u= pd.read_csv("X_2u_chr9.csv")
X_2d= pd.read_csv("X_2d_chr9.csv")
X_2d1= pd.read_csv("X_2d1_chr9.csv")
X_2d2= pd.read_csv("X_2d2_chr9.csv")
# pca
principalupstreamwinnum= pd.read_csv("principalupstreamwinnum_chr9")
principaldownstreamwinnum= pd.read_csv("principaldownstreamwinnum_chr9")
#full matrix
dfDownwinnum= pd.read_csv("dfDownwinnum_chr9")
dfUpwinnum= pd.read_csv("dfUpwinnum_chr9")
Upstreamdropwin = pd.read_csv("Upstreamdropwin_chr9.csv")
Downstreamdropwin = pd.read_csv("Downstreamdropwin_chr9")
#louvain
upresult= pd.read_csv("upresult_chr9.tsv")
downresult= pd.read_csv("downresult_chr9.tsv")
dfupprediction= pd.read_csv("dfupprediction_chr9.tsv")
dfdownprediction= pd.read_csv("dfdownprediction_chr9.tsv")
# save the dfs
dfsumTRpltup= pd.read_csv("dfsumTRpltup_chr9.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chr9.tsv")
umap2dimensionup_chr9= pd.read_csv("umap2dimensionup_chr9.tsv")
umap2dimensiondown_chr9= pd.read_csv("umap2dimensiondown_chr9.tsv")
# reduced 
reductdownumap2d= pd.read_csv("reductdownumap2d_chr9.tsv")
reductupumap2d_chr9= pd.read_csv("reductupumap2d_chr9.tsv")
# louvain byw
dfdownpredictionbyw= pd.read_csv("downpredictionbyw_chr9.tsv")
dfuppredictionbyw= pd.read_csv("uppredictionbyw_chr9.tsv")

upresult= pd.read_csv("upresult_chr9.tsv")
downresult= pd.read_csv("downresult_chr9.tsv")


# In[49]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

mostimportant10= most_important10_differentmethod[:,0]


Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns

up_mostimportantsubset=Upstreamdropwin[mostimportant10]
up_mostimportantsubset

importantupresult = pd.concat([up_mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']

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



# In[50]:




#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

up_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

up_mostimportant10= up_most_important10_differentmethod[:,0]
up_mostimportant10= list(up_mostimportant10)

mostimportantsubset=Upstreamdropwin[up_mostimportant10]
mostimportantsubset

importantupresult = pd.concat([mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']
 
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
                          
down_most_important10_differentmethod
Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 
down_mostimportantsubset=Downstreamdropwin[down_mostimportant10]
down_mostimportantsubset
importantdownresult = pd.concat([down_mostimportantsubset, dfdownprediction], axis=1)
importantdownresult.columns = [*importantdownresult .columns[:-1], 'p']

importantdownresult
importantupresult
meanupimportant = importantupresult.groupby('p').mean()
meanupdf=pd.DataFrame(meanupimportant)

print(meanupdf)
                          
                          
meandownimportant = importantdownresult.groupby('p').mean()
meandowndf=pd.DataFrame(meandownimportant)
meandowndf


countdown = importantdownresult.value_counts('p')
countup =importantupresult.value_counts('p')
dfcountup=pd.DataFrame(countup)
dfcountdown=pd.DataFrame(countdown)

dfcountupwhere = np.where((dfcountup[0]<10),False,True)
dfcountdownwhere = np.where((dfcountdown[0]<10),False,True)

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


# In[51]:


import pandas as pd

dfsumTRpltdown = np.log10(dfsumTRpltdown)
dfsumTRpltdown
dfsumTRpltup = np.log10(dfsumTRpltup)
dfsumTRpltup


# In[52]:


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
plt.title('most important 10 for upstream 60Seqclassdf_HeatMap_chr9')
plt.savefig('upstream_mostimportant10_HeatMap_60seqclass_chr9.pdf', dpi=299)
sns.clustermap(upheatsum60)


# In[53]:


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
plt.title('most important 10 for downstream 60Seqclassdf_HeatMap_chr9')
plt.savefig('downstream_mostimportant10_HeatMap_60seqclass_chr9.pdf', dpi=299)
sns.clustermap(downheatsum60)


# In[54]:


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
plt.title('upstream zscore_HeatMap_chr9')
plt.savefig('upstream_zscore__heatmap_chr9.pdf', dpi=299)
sns.heatmap(meanimportantzscoreup60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreup60)


# In[55]:


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
plt.title('downstream zscore_HeatMa_chr9')
plt.savefig('downstream_zscore__heatmap_chr9.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60)


# In[56]:


reductupumap2d_chr9['p'] = dfuppredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='p',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


# In[57]:


dfUpseqname = dfUpwinnum.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6','49':'losig1','50':'losig2','51':'losig3','52':'losig4'})
dfUpseqname
dfUpseqname=dfUpseqname.drop(columns=['TR_id', 'Win_num','Result'])

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
plt.title('upstream zscore_HeatMap_chr9')
plt.savefig('upstream_zscore__heatmap_chr9.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredup60seq)


# In[58]:


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
plt.title('downstream zscore_HeatMap_chr9')
plt.savefig('downstream_zscore__heatmap_chr9.pdf', dpi=299)
sns.heatmap(meanimportantzscoreddown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreddown60seq)


# In[ ]:





# In[59]:


dfsumTRpltup= pd.read_csv("dfsumTRpltup_chr9.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chr9.tsv")

dfUpseqname = dfsumTRpltup.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfUpseqname

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
plt.title('upstream zscore_HeatMap_chr9')
plt.savefig('upstream_zscore__heatmap_chr9.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.set(rc={'figure.figsize':(15.7,8.27)})
sns.clustermap(meanimportantzscoredup60seq, square=False,figsize=(15.7,8.27), xticklabels=1)



dfDownseqname = dfsumTRpltdown.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfDownseqname

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
plt.title('downstream zscore_HeatMap_chr9')
plt.savefig('downstream_zscore__heatmap_chr9.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60seq, square=False,figsize=(15.7,8.27), xticklabels=1)

reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()

reductupumap2d_chr9['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chr9['CTCF_high'] = np.where(reductupumap2d_chr9['CTCF']>10,True,False)
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>6,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>9,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()

reductupumap2d_chr9['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chr9['CTCF_high'] = np.where(reductupumap2d_chr9['CTCF']>6,True,False)
sns.scatterplot(data=reductupumap2d_chr9, x=reductupumap2d_chr9['0'], y=reductupumap2d_chr9['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>9,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


# In[60]:


dfUpseqname = dfsumTRpltup.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfUpseqname

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
plt.title('upstream zscore_HeatMap_chr9')
plt.savefig('upstream_zscore__heatmap_chr9.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.set(rc={'figure.figsize':(15.7,8.27)})
sns.clustermap(meanimportantzscoredup60seq, square=False,figsize=(15.7,8.27), xticklabels=1)


# In[61]:


plt.hist(reductdownumap2d['CTCF'])


# In[62]:


plt.hist(reductdownumap2d['P'])


# # chrx30_5kb

# In[63]:


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
X_2u2=  pd.read_csv("X_2u2_chrx30_5kb.csv")
X_2u1= pd.read_csv("X_2u1_chrx30_5kb.csv")
X_2u= pd.read_csv("X_2u_chrx30_5kb.csv")
X_2d= pd.read_csv("X_2d_chrx30_5kb.csv")
X_2d1= pd.read_csv("X_2d1_chrx30_5kb.csv")
X_2d2= pd.read_csv("X_2d2_chrx30_5kb.csv")
# pca
principalupstreamwinnum= pd.read_csv("principalupstreamwinnum_chrx30_5kb")
principaldownstreamwinnum= pd.read_csv("principaldownstreamwinnum_chrx30_5kb")
#full matrix
dfDownwinnum= pd.read_csv("dfDownwinnum_chrx30_5kb")
dfUpwinnum= pd.read_csv("dfUpwinnum_chrx30_5kb")
Upstreamdropwin = pd.read_csv("Upstreamdropwin_chrx30_5kb.csv")
Downstreamdropwin = pd.read_csv("Downstreamdropwin_chrx30_5kb")
#louvain
upresult= pd.read_csv("upresult_chrx30_5kb.tsv")
downresult= pd.read_csv("downresult_chrx30_5kb.tsv")
dfupprediction= pd.read_csv("dfupprediction_chrx30_5kb.tsv")
dfdownprediction= pd.read_csv("dfdownprediction_chrx30_5kb.tsv")
# save the dfs
dfsumTRpltup= pd.read_csv("dfsumTRpltup_chrx30_5kb.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chrx30_5kb.tsv")
umap2dimensionup_chrx= pd.read_csv("umap2dimensionup_chrx30_5kb.tsv")
umap2dimensiondown_chrx= pd.read_csv("umap2dimensiondown_chrx30_5kb.tsv")
# reduced 
reductdownumap2d= pd.read_csv("reductdownumap2d_chrx30_5kb.tsv")
reductupumap2d_chrx= pd.read_csv("reductupumap2d_chrx30_5kb.tsv")
# louvain byw
dfdownpredictionbyw= pd.read_csv("downpredictionbyw_chrx30_5kb.tsv")
dfuppredictionbyw= pd.read_csv("uppredictionbyw_chrx30_5kb.tsv")

upresult= pd.read_csv("upresult_chrx30_5kb.tsv")
downresult= pd.read_csv("downresult_chrx30_5kb.tsv")


# In[64]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

mostimportant10= most_important10_differentmethod[:,0]


Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns

up_mostimportantsubset=Upstreamdropwin[mostimportant10]
up_mostimportantsubset

importantupresult = pd.concat([up_mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']

from sklearn.decomposition import PCA
pcadfupstreamMatreal = PCA(n_components=2)
principalComponentsdfupstreamMatreal = pcadfupstreamMatreal.fit_transform(Upstreamdropwin)

plt.bar(x=range(2), height= pcadfupstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfupstreamMatreal.explained_variance_ratio_)

principalupstreamDfreal = pd.DataFrame (data = principalComponentsdfupstreamMatreal, columns = ['a', 'b'])
plt.scatter(principalupstreamDfreal['a'], principalupstreamDfreal['b'], c='purple')
plt.title('most important 10 for upstream full Matrix')
plt.savefig('upstream_mostimportant10_chrx30_5kb.pdf', dpi=299)
plt.show()

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
plt.savefig('downstream_mostimportant10_chx30_5kb.pdf', dpi=299)
plt.show()




#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

up_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

up_mostimportant10= up_most_important10_differentmethod[:,0]
up_mostimportant10= list(up_mostimportant10)

mostimportantsubset=Upstreamdropwin[up_mostimportant10]
mostimportantsubset

importantupresult = pd.concat([mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']
 
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
                          
down_most_important10_differentmethod
Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 
down_mostimportantsubset=Downstreamdropwin[down_mostimportant10]
down_mostimportantsubset
importantdownresult = pd.concat([down_mostimportantsubset, dfdownprediction], axis=1)
importantdownresult.columns = [*importantdownresult .columns[:-1], 'p']

importantdownresult
importantupresult
meanupimportant = importantupresult.groupby('p').mean()
meanupdf=pd.DataFrame(meanupimportant)

print(meanupdf)
                          
                          
meandownimportant = importantdownresult.groupby('p').mean()
meandowndf=pd.DataFrame(meandownimportant)
meandowndf


countdown = importantdownresult.value_counts('p')
countup =importantupresult.value_counts('p')
dfcountup=pd.DataFrame(countup)
dfcountdown=pd.DataFrame(countdown)

dfcountupwhere = np.where((dfcountup[0]<10),False,True)
dfcountdownwhere = np.where((dfcountdown[0]<10),False,True)

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


# In[65]:



import pandas as pd

dfsumTRpltdown = np.log10(dfsumTRpltdown)
dfsumTRpltdown
dfsumTRpltup = np.log10(dfsumTRpltup)
dfsumTRpltup


# In[66]:



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
plt.title('most important 10 for upstream 60Seqclassdf_HeatMap_chrx30_5kb')
plt.savefig('upstream_mostimportant10_HeatMap_60seqclass_chrx30_5kb.pdf', dpi=299)
sns.clustermap(upheatsum60)

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
plt.title('most important 10 for downstream 60Seqclassdf_HeatMap_chrx30_5kb')
plt.savefig('downstream_mostimportant10_HeatMap_60seqclass_chrx30_5kb.pdf', dpi=299)
sns.clustermap(downheatsum60)


# In[67]:


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
plt.title('upstream zscore_HeatMap_chrx30_5kb')
plt.savefig('upstream_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoreup60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreup60)

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
plt.title('downstream zscore_HeatMa_chrx30_5kb')
plt.savefig('downstream_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60)


# In[68]:


reductupumap2d_chrx['p'] = dfuppredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d_chrx, x=reductupumap2d_chrx['0'], y=reductupumap2d_chrx['1'], hue='p',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


# In[69]:



dfUpseqname = dfUpwinnum.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6','49':'losig1','50':'losig2','51':'losig3','52':'losig4'})
dfUpseqname
dfUpseqname=dfUpseqname.drop(columns=['TR_id', 'Win_num','Result'])

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
plt.title('upstream zscore_HeatMap_chrx30_5kb')
plt.savefig('upstream_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredup60seq)

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
plt.title('downstream zscore_HeatMap_chrx30_5kb')
plt.savefig('downstream_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoreddown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreddown60seq)


# In[70]:


dfsumTRpltup= pd.read_csv("dfsumTRpltup_chrx30_5kb.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chrx30_5kb.tsv")

dfUpseqname = dfsumTRpltup.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfUpseqname

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
plt.title('upstream zscore_HeatMap_chrx30_5kb')
plt.savefig('upstream_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.set(rc={'figure.figsize':(15.7,8.27)})
sns.clustermap(meanimportantzscoredup60seq,square=False,figsize=(15.7,8.27), xticklabels=1)
sns.set(rc={'figure.figsize':(15.7,8.27)})


# In[71]:


dfDownseqname = dfsumTRpltdown.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6','41':'losig1','42':'losig2','43':'losig3','44':'losig4'})
dfDownseqname

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
plt.title('downstream zscore_HeatMap_chrx30_5kb')
plt.savefig('downstream_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60seq, square=False,figsize=(15.7,8.27), xticklabels=1)


# In[72]:




reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()

reductupumap2d_chrx['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chrx['CTCF_high'] = np.where(reductupumap2d_chrx['CTCF']>10,True,False)
sns.scatterplot(data=reductupumap2d_chrx, x=reductupumap2d_chrx['0'], y=reductupumap2d_chrx['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()



reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()

reductupumap2d_chrx['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chrx['CTCF_high'] = np.where(reductupumap2d_chrx['CTCF']>10,True,False)
sns.scatterplot(data=reductupumap2d_chrx, x=reductupumap2d_chrx['0'], y=reductupumap2d_chrx['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


# # chrx30

# In[73]:


# tsne
X_2u2=  pd.read_csv("X_2u2_chrx30.csv")
X_2u1= pd.read_csv("X_2u1_chrx30.csv")
X_2u= pd.read_csv("X_2u_chrx30.csv")

# pca
principalupstreamwinnum= pd.read_csv("principalupstreamwinnum_chrx30")
principaldownstreamwinnum= pd.read_csv("principaldownstreamwinnum_chrx30")
#full matrix
dfDownwinnum= pd.read_csv("dfDownwinnum_chrx30")
dfUpwinnum= pd.read_csv("dfUpwinnum_chrx30")
Upstreamdropwin = pd.read_csv("Upstreamdropwin_chrx30.csv")
Downstreamdropwin = pd.read_csv("Downstreamdropwin_chrx30")
#louvain
upresult= pd.read_csv("upresult_chrx30.tsv")
downresult= pd.read_csv("downresult_chrx30.tsv")
dfupprediction= pd.read_csv("dfupprediction_chrx30.tsv")
dfdownprediction= pd.read_csv("dfdownprediction_chrx30.tsv")
# save the dfs
dfsumTRpltup= pd.read_csv("dfsumTRpltup_chrx30.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chrx30.tsv")
umap2dimensionup_chrx= pd.read_csv("umap2dimensionup_chrx30.tsv")
umap2dimensiondown_chrx= pd.read_csv("umap2dimensiondown_chrx30.tsv")
# reduced 
reductdownumap2d= pd.read_csv("reductdownumap2d_chrx30.tsv")
reductupumap2d_chrx= pd.read_csv("reductupumap2d_chrx30.tsv")
# louvain byw
dfdownpredictionbyw= pd.read_csv("downpredictionbyw_chrx30.tsv")
dfuppredictionbyw= pd.read_csv("uppredictionbyw_chrx30.tsv")


# In[74]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

mostimportant10= most_important10_differentmethod[:,0]


Upstreamdropwin.columns = Upstreamdropwin.columns.astype(int) 
Upstreamdropwin.columns

up_mostimportantsubset=Upstreamdropwin[mostimportant10]
up_mostimportantsubset

importantupresult = pd.concat([up_mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']

from sklearn.decomposition import PCA
pcadfupstreamMatreal = PCA(n_components=2)
principalComponentsdfupstreamMatreal = pcadfupstreamMatreal.fit_transform(Upstreamdropwin)

plt.bar(x=range(2), height= pcadfupstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfupstreamMatreal.explained_variance_ratio_)

principalupstreamDfreal = pd.DataFrame (data = principalComponentsdfupstreamMatreal, columns = ['a', 'b'])
plt.scatter(principalupstreamDfreal['a'], principalupstreamDfreal['b'], c='purple')
plt.title('most important 10 for upstream full Matrix')
plt.savefig('upstream_mostimportant10_chrx30.pdf', dpi=299)
plt.show()

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
plt.savefig('downstream_mostimportant10_chx30.pdf', dpi=299)
plt.show()




#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(Upstreamdropwin)

up_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

up_mostimportant10= up_most_important10_differentmethod[:,0]
up_mostimportant10= list(up_mostimportant10)

mostimportantsubset=Upstreamdropwin[up_mostimportant10]
mostimportantsubset

importantupresult = pd.concat([mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']
 
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
                          
down_most_important10_differentmethod
Downstreamdropwin.columns = Downstreamdropwin.columns.astype(int) 
Downstreamdropwin.columns 
down_mostimportantsubset=Downstreamdropwin[down_mostimportant10]
down_mostimportantsubset
importantdownresult = pd.concat([down_mostimportantsubset, dfdownprediction], axis=1)
importantdownresult.columns = [*importantdownresult .columns[:-1], 'p']

importantdownresult
importantupresult
meanupimportant = importantupresult.groupby('p').mean()
meanupdf=pd.DataFrame(meanupimportant)

print(meanupdf)
                          
                          
meandownimportant = importantdownresult.groupby('p').mean()
meandowndf=pd.DataFrame(meandownimportant)
meandowndf


countdown = importantdownresult.value_counts('p')
countup =importantupresult.value_counts('p')
dfcountup=pd.DataFrame(countup)
dfcountdown=pd.DataFrame(countdown)

dfcountupwhere = np.where((dfcountup[0]<10),False,True)
dfcountdownwhere = np.where((dfcountdown[0]<10),False,True)

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


# # filteration

# below here, we only see chr9_30winnum 5kb and chrx_30winnum 5kb for the further analysis
# 1) drop all the rows that scores are a below the threshold for the mean
# 2) drop all the columns in seq class scores after 40s (2% of the genome, low signals)

# ChrX_5kb

# In[75]:


# chrx 30_5kb first

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
X_2u2_chrx=  pd.read_csv("X_2u2_chrx30_5kb.csv")
X_2u1_chrx= pd.read_csv("X_2u1_chrx30_5kb.csv")
X_2u_chrx= pd.read_csv("X_2u_chrx30_5kb.csv")
X_2d_chrx= pd.read_csv("X_2d_chrx30_5kb.csv")
X_2d1_chrx= pd.read_csv("X_2d1_chrx30_5kb.csv")
X_2d2_chrx= pd.read_csv("X_2d2_chrx30_5kb.csv")
# pca
principalupstreamwinnum_chrx= pd.read_csv("principalupstreamwinnum_chrx30_5kb")
principaldownstreamwinnum_chrx= pd.read_csv("principaldownstreamwinnum_chrx30_5kb")
#full matrix
dfDownwinnum_chrx= pd.read_csv("dfDownwinnum_chrx30_5kb")
dfUpwinnum_chrx= pd.read_csv("dfUpwinnum_chrx30_5kb")
Upstreamdropwin_chrx = pd.read_csv("Upstreamdropwin_chrx30_5kb.csv")
Downstreamdropwin_chrx = pd.read_csv("Downstreamdropwin_chrx30_5kb")
#louvain
upresult_chrx= pd.read_csv("upresult_chrx30_5kb.tsv")
downresult_chrx= pd.read_csv("downresult_chrx30_5kb.tsv")
dfupprediction_chrx= pd.read_csv("dfupprediction_chrx30_5kb.tsv")
dfdownprediction_chrx= pd.read_csv("dfdownprediction_chrx30_5kb.tsv")
# save the dfs
dfsumTRpltup_chrx= pd.read_csv("dfsumTRpltup_chrx30_5kb.tsv")
dfsumTRpltdown_chrx= pd.read_csv("dfsumTRpltdown_chrx30_5kb.tsv")
umap2dimensionup_chrx= pd.read_csv("umap2dimensionup_chrx30_5kb.tsv")
umap2dimensiondown_chrx= pd.read_csv("umap2dimensiondown_chrx30_5kb.tsv")
# reduced 
reductdownumap2d_chrx= pd.read_csv("reductdownumap2d_chrx30_5kb.tsv")
reductupumap2d_chrx= pd.read_csv("reductupumap2d_chrx30_5kb.tsv")
# louvain byw
dfdownpredictionbyw_chrx= pd.read_csv("downpredictionbyw_chrx30_5kb.tsv")
dfuppredictionbyw_chrx= pd.read_csv("uppredictionbyw_chrx30_5kb.tsv")

upresult_chrx= pd.read_csv("upresult_chrx30_5kb.tsv")
downresult_chrx= pd.read_csv("downresult_chrx30_5kb.tsv")


# In[76]:


Upstreamdropwin_chrx


# In[77]:


Downstreamdropwin_chrx


# In[78]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfUpwinnum_chrx = df.loc[df["Result"] == (tr_id)].copy()
    sumTRUP_chrx =dfUpwinnum_chrx[str(9+seqclass)].sum()
    
    return sumTRUP_chrx

colsUP = list()
colsUP = dfUpwinnum_chrx.columns.tolist()
print(type(colsUP))
print(len(colsUP))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRUP_chrx= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfUpwinnum_chrx["Result"].tolist())):
    for key in translater.keys():
        sumTRUP_chrx[key].append(sum_df(dfUpwinnum_chrx,i, translater[key]))

dfUpwinnumdrop_chrx = dfUpwinnum_chrx


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfDownwinnum_chrx = df.loc[df["Result"] == (tr_id)].copy()
    sumTRDOWN_chrx =dfDownwinnum_chrx[str(9+seqclass)].sum()
    
    return sumTRDOWN_chrx



colsDown = list()
colsDown = dfDownwinnum_chrx.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRDOWN_chrx= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfDownwinnum_chrx["Result"].tolist())):
    for key in translater.keys():
        sumTRDOWN_chrx[key].append(sum_df(dfDownwinnum_chrx,i, translater[key]))

dfDownwinnumdrop_chrx = dfDownwinnum_chrx



# In[79]:


# upstream
meanupstream=Upstreamdropwin_chrx.mean(axis=1)
meanupstream


# In[80]:


meandownstream = Downstreamdropwin_chrx.mean(axis = 1)
meandownstream


# In[81]:


meandownstream.hist


# In[82]:


# upstream
meanupstream_row=Upstreamdropwin_chrx.mean(axis=0)
meanupstream_row


# In[83]:


# downstream
meandownstream_row=Downstreamdropwin_chrx.mean(axis=0)
meandownstream_row


# In[84]:


meandowndf =pd.concat([Downstreamdropwin_chrx,meandownstream], axis=1, join = 'inner')
meandowndf.columns = [*meandowndf.columns[:-1], 'mean']
meandowndf


# In[85]:


meandowndf.columns = [*meandowndf.columns[:-1], 'mean']
meandowndf


# In[86]:



meanupdf =pd.concat([Upstreamdropwin_chrx,meanupstream], axis=1, join = 'inner')
meanupdf.columns = [*meanupdf.columns[:-1], 'mean']
meanupdf


# In[87]:




reductdownumap2d_chrx['mean'] = meandowndf['mean']
plt.hist(reductdownumap2d_chrx['mean'])


# In[88]:



reductupumap2d_chrx['mean'] = meanupdf['mean']
plt.hist(reductupumap2d_chrx['mean'])


# In[89]:


reductdownumap2d_chrx['mean'] = meandowndf['mean']
reductdownumap2d_chrx['mean_high'] = np.where(reductdownumap2d_chrx['mean']>0.0,True,False)
sns.scatterplot(data=reductdownumap2d_chrx, x=reductdownumap2d_chrx['0'], y=reductdownumap2d_chrx['1'], hue='mean_high',palette="tab10")
plt.show()


# In[90]:


reductupumap2d_chrx['mean'] = meanupdf['mean']
reductupumap2d_chrx['mean_high'] = np.where(reductupumap2d_chrx['mean']>0.0,True,False)
sns.scatterplot(data=reductupumap2d_chrx, x=reductupumap2d_chrx['0'], y=reductupumap2d_chrx['1'], hue='mean_high',palette="tab10")
plt.show()


# In[91]:


meandfup1 = meanupdf[meanupdf['mean'] > -1]  
meandfup1


# In[92]:



meandfdown1 = meandowndf[meandowndf['mean'] > -1]  
meandfdown1


# In[93]:



dfsumTRUP=pd.DataFrame(sumTRUP_chrx)
dfsumTRDOWN=pd.DataFrame(sumTRDOWN_chrx)


# In[94]:


sumdfdown1 = dfsumTRDOWN[meandowndf['mean'] > -1]  
sumdfdown1


# In[95]:



sumdfup1 = dfsumTRUP[meanupdf['mean'] > -1] 
sumdfup1


# In[96]:



meandfdown1.to_csv("meandfdown1_chrx30_5kb.tsv")
meandfup1.to_csv("meandfup1_chrx30_5kb.tsv")


# In[97]:



meandfdown1 = meandfdown1.drop(['mean'],axis =1)
meandfup1 = meandfup1.drop(['mean'],axis =1)


# PCA on the dropped row df

# In[98]:


meandfup1


# In[99]:



# downstream 
x = meandfdown1.values
y = meandfdown1.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfdowndrop = PCA(n_components=8)
pcadfdowndropmat = pcadfdowndrop.fit_transform(x)

plt.bar(x=range(8), height= pcadfdowndrop.explained_variance_ratio_)
plt.show()

sum(pcadfdowndrop.explained_variance_ratio_)


principaldownmat = pd.DataFrame (data = pcadfdowndropmat, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principaldownmat['a'], principaldownmat['b'], c='green')
plt.show()


# In[100]:




#upstream 

x = meandfup1.values
y = meandfup1.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfupdrop = PCA(n_components=8)
pcadfupdropmat = pcadfupdrop.fit_transform(x)

plt.bar(x=range(8), height= pcadfupdrop.explained_variance_ratio_)
plt.show()

sum(pcadfupdrop.explained_variance_ratio_)


principalupmat = pd.DataFrame (data = pcadfupdropmat, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalupmat['a'], principalupmat['b'], c='green')
plt.show()


# In[101]:




principalupmat.to_csv("principalupmat_chrx30_5kb", index= None)
principaldownmat.to_csv("principaldownmat_chrx30_5kb", index= None)


# umap on the dropped row df full matrix

# In[102]:


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


# In[103]:




# downstream fullmatrix
meandfdown1
x = meandfdown1.values
y = meandfdown1.values
x = StandardScaler().fit_transform(x)

meandfdown1umap = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
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
meandfdown1umapfull = meandfdown1umap.fit_transform(x)

# Check the shape of the new data
print('Shape of X_trans: ',meandfdown1umapfull.shape)
meandfdown1umapfull=pd.DataFrame(meandfdown1umapfull)

plt.scatter(x=meandfdown1umapfull[0],y=meandfdown1umapfull[1])
plt.show()


## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=meandfdown1umapfull,x= meandfdown1umapfull[0],y=meandfdown1umapfull[1], hue=meandfdown1umapfull[0])


# In[104]:




# upstream fullmatrix
meandfup1
x = meandfup1.values
y = meandfup1.values
x = StandardScaler().fit_transform(x)

meandfup1umap = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
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
meandfup1umapfull = meandfup1umap.fit_transform(x)

# Check the shape of the new data
print('Shape of X_trans: ',meandfup1umapfull.shape)
meandfup1umapfull=pd.DataFrame(meandfup1umapfull)

plt.scatter(x=meandfup1umapfull[0],y=meandfup1umapfull[1])
plt.show()


## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=meandfup1umapfull,x= meandfup1umapfull[0],y=meandfup1umapfull[1], hue=meandfup1umapfull[0])


# In[105]:


dfsumTRUP


# In[106]:


meandfup1umapfull


# In[107]:


umap2dimensionup_chrx


# In[108]:


sumtrupseries = dfsumTRUP.squeeze()


# In[109]:


sumtrdownseries = dfsumTRDOWN.squeeze()


# In[110]:


for key in translater.keys():
    meandfup1umapfull[key] = sumdfup1[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=meandfup1umapfull, x= 0, y= 1, hue=key)
    plt.show()


# In[111]:


for key in translater.keys():
    meandfdown1umapfull[key] = sumdfdown1[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=meandfdown1umapfull, x= 0, y= 1, hue=key)
    plt.show()


# In[112]:


for key in translater.keys():

    principalupmat[key] = sumdfup1[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupmat, x="a", y="b", hue=key)
    plt.show()


# In[113]:


for key in translater.keys():

    principaldownmat[key] = sumdfdown1[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownmat, x="a", y="b", hue=key)
    plt.show()


# chr9_5kb

# In[2]:


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
X_2u2_chr9=  pd.read_csv("X_2u2_chr930_5kb.csv")
X_2u1_chr9= pd.read_csv("X_2u1_chr930_5kb.csv")
X_2u_chr9= pd.read_csv("X_2u_chr930_5kb.csv")
X_2d_chr9= pd.read_csv("X_2d_chr930_5kb.csv")
X_2d1_chr9= pd.read_csv("X_2d1_chr930_5kb.csv")
X_2d2_chr9= pd.read_csv("X_2d2_chr930_5kb.csv")
# pca
principalupstreamwinnum_chr9= pd.read_csv("principalupstreamwinnum_chr930_5kb")
principaldownstreamwinnum_chr9= pd.read_csv("principaldownstreamwinnum_chr930_5kb")
#full matrix
dfDownwinnum_chr9= pd.read_csv("dfDownwinnum_chr930_5kb")
dfUpwinnum_chr9= pd.read_csv("dfUpwinnum_chr930_5kb")
Upstreamdropwin_chr9 = pd.read_csv("Upstreamdropwin_chr930_5kb.csv")
Downstreamdropwin_chr9 = pd.read_csv("Downstreamdropwin_chr930_5kb")
#louvain
upresult_chr9= pd.read_csv("upresult_chr930_5kb.tsv")
downresult_chr9= pd.read_csv("downresult_chr930_5kb.tsv")
dfupprediction_chr9= pd.read_csv("dfupprediction_chr930_5kb.tsv")
dfdownprediction_chr9= pd.read_csv("dfdownprediction_chr930_5kb.tsv")
# save the dfs
dfsumTRpltup_chr9= pd.read_csv("dfsumTRpltup_chr930_5kb.tsv")
dfsumTRpltdown_chr9= pd.read_csv("dfsumTRpltdown_chr930_5kb.tsv")
umap2dimensionup_chr9= pd.read_csv("umap2dimensionup_chr930_5kb.tsv")
umap2dimensiondown_chr9= pd.read_csv("umap2dimensiondown_chr930_5kb.tsv")
# reduced 
reductdownumap2d_chr9= pd.read_csv("reductdownumap2d_chr930_5kb.tsv")
reductupumap2d_chr9= pd.read_csv("reductupumap2d_chr930_5kb.tsv")
# louvain byw
dfdownpredictionbyw_chr9= pd.read_csv("downpredictionbyw_chr930_5kb.tsv")
dfuppredictionbyw_chr9= pd.read_csv("uppredictionbyw_chr930_5kb.tsv")

upresult_chr9= pd.read_csv("upresult_chr930_5kb.tsv")
downresult_chr9= pd.read_csv("downresult_chr930_5kb.tsv")


# In[4]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfUpwinnum_chr9 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRUP_chr9 =dfUpwinnum_chr9[str(9+seqclass)].sum()
    
    return sumTRUP_chr9

colsUP = list()
colsUP = dfUpwinnum_chr9.columns.tolist()
print(type(colsUP))
print(len(colsUP))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRUP_chr9= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfUpwinnum_chr9["Result"].tolist())):
    for key in translater.keys():
        sumTRUP_chr9[key].append(sum_df(dfUpwinnum_chr9,i, translater[key]))

dfUpwinnumdrop_chr9 = dfUpwinnum_chr9


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfDownwinnum_chr9 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRDOWN_chr9 =dfDownwinnum_chr9[str(9+seqclass)].sum()
    
    return sumTRDOWN_chr9



colsDown = list()
colsDown = dfDownwinnum_chr9.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRDOWN_chr9= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfDownwinnum_chr9["Result"].tolist())):
    for key in translater.keys():
        sumTRDOWN_chr9[key].append(sum_df(dfDownwinnum_chr9,i, translater[key]))

dfDownwinnumdrop_chr9 = dfDownwinnum_chr9




# upstream
meanupstream=Upstreamdropwin_chr9.mean(axis=1)
meanupstream

meandownstream = Downstreamdropwin_chr9.mean(axis = 1)
meandownstream


# In[5]:


# upstream
meanupstream=Upstreamdropwin_chr9.mean(axis=1)
meanupstream

meandownstream = Downstreamdropwin_chr9.mean(axis = 1)
meandownstream


meandowndf =pd.concat([Downstreamdropwin_chr9,meandownstream], axis=1, join = 'inner')
meandowndf.columns = [*meandowndf.columns[:-1], 'mean']
meandowndf

meanupdf =pd.concat([Upstreamdropwin_chr9,meanupstream], axis=1, join = 'inner')
meanupdf.columns = [*meanupdf.columns[:-1], 'mean']
meanupdf

reductdownumap2d_chr9['mean'] = meandowndf['mean']
plt.hist(reductdownumap2d_chr9['mean'])

reductupumap2d_chr9['mean'] = meanupdf['mean']
plt.hist(reductupumap2d_chr9['mean'])


# In[8]:




meandfup1 = meanupdf[meanupdf['mean'] > -1]  
meandfup1

meandfdown1 = meandowndf[meandowndf['mean'] > -1]  
meandfdown1

dfsumTRUP=pd.DataFrame(sumTRUP_chr9)
dfsumTRDOWN=pd.DataFrame(sumTRDOWN_chr9)

sumdfdown1 = dfsumTRDOWN[meandowndf['mean'] > -1]  
sumdfdown1

sumdfup1 = dfsumTRUP[meanupdf['mean'] > -1] 
sumdfup1

meandfdown1.to_csv("meandfdown1_chr930_5kb.tsv")
meandfup1.to_csv("meandfup1_chr930_5kb.tsv")

meandfdown1 = meandfdown1.drop(['mean'],axis =1)
meandfup1 = meandfup1.drop(['mean'],axis =1)
# downstream 
x = meandfdown1.values
y = meandfdown1.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfdowndrop = PCA(n_components=8)
pcadfdowndropmat = pcadfdowndrop.fit_transform(x)

plt.bar(x=range(8), height= pcadfdowndrop.explained_variance_ratio_)
plt.show()

sum(pcadfdowndrop.explained_variance_ratio_)


principaldownmat = pd.DataFrame (data = pcadfdowndropmat, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principaldownmat['a'], principaldownmat['b'], c='green')
plt.show()


#upstream 

x = meandfup1.values
y = meandfup1.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfupdrop = PCA(n_components=8)
pcadfupdropmat = pcadfupdrop.fit_transform(x)

plt.bar(x=range(8), height= pcadfupdrop.explained_variance_ratio_)
plt.show()

sum(pcadfupdrop.explained_variance_ratio_)


principalupmat = pd.DataFrame (data = pcadfupdropmat, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalupmat['a'], principalupmat['b'], c='green')
plt.show()



principalupmat.to_csv("principalupmat_chr930_5kb", index= None)
principaldownmat.to_csv("principaldownmat_chr930_5kb", index= None)


# In[9]:



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
meandfdown1
x = meandfdown1.values
y = meandfdown1.values
x = StandardScaler().fit_transform(x)

meandfdown1umap = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
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
meandfdown1umapfull = meandfdown1umap.fit_transform(x)

# Check the shape of the new data
print('Shape of X_trans: ',meandfdown1umapfull.shape)
meandfdown1umapfull=pd.DataFrame(meandfdown1umapfull)

plt.scatter(x=meandfdown1umapfull[0],y=meandfdown1umapfull[1])
plt.show()


## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=meandfdown1umapfull,x= meandfdown1umapfull[0],y=meandfdown1umapfull[1], hue=meandfdown1umapfull[0])



# upstream fullmatrix
meandfup1
x = meandfup1.values
y = meandfup1.values
x = StandardScaler().fit_transform(x)

meandfup1umap = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
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
meandfup1umapfull = meandfup1umap.fit_transform(x)

# Check the shape of the new data
print('Shape of X_trans: ',meandfup1umapfull.shape)
meandfup1umapfull=pd.DataFrame(meandfup1umapfull)

plt.scatter(x=meandfup1umapfull[0],y=meandfup1umapfull[1])
plt.show()


## hue corresponds to the value
## write the function that would do everything 
## dark - most similar features 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=meandfup1umapfull,x= meandfup1umapfull[0],y=meandfup1umapfull[1], hue=meandfup1umapfull[0])


# In[10]:


for key in translater.keys():
    meandfup1umapfull[key] = sumdfup1[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=meandfup1umapfull, x= 0, y= 1, hue=key)
    plt.show()


# In[11]:


for key in translater.keys():
    meandfdown1umapfull[key] = sumdfdown1[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=meandfdown1umapfull, x= 0, y= 1, hue=key)
    plt.show()


# In[12]:


for key in translater.keys():

    principalupmat[key] = sumdfup1[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupmat, x="a", y="b", hue=key)
    plt.show()


# In[13]:


for key in translater.keys():

    principaldownmat[key] = sumdfup1[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownmat, x="a", y="b", hue=key)
    plt.show()


# # Drop all columns in seqclss after 40

# # chrx30

# In[1]:


import pandas as pd
import numpy as np


df = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_subtyping/Sei/chrX30/chromatin-profiles-hdf5/chrx4colnoN_30_row_labels.txt", sep="\t",low_memory=False)
df


dfSei = np.load("/data/projects/nanopore/RepeatExpansion/TR_subtyping/chrX30/chrx4colnoN_30.ref.raw_sequence_class_scores.npy")
dfSei = pd.DataFrame(dfSei)
dfSei


# concat axis default =0
dfinput = pd.concat([df,dfSei], axis = 1)
display(dfinput)


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


# In[ ]:





# In[2]:


# downstream 
dfDownstreamdropwinnum=(dfDownstreamdropwinnum.reset_index(drop=True))
dfDownwinnum = dfDownstreamdropwinnum.drop(columns=['chromosome','1','window','basepair','4','5','6','7','6','7'])
print(dfDownwinnum)
# upstream
dfUpstreamdropwinnum=(dfUpstreamdropwinnum.reset_index(drop=True))
dfUpwinnum = dfUpstreamdropwinnum.drop(columns=['chromosome','1','window','basepair','4','5','6','7','6','7'])
print(dfUpwinnum)


# In[3]:


print(dfDownwinnum)


# In[4]:


dfdropdown40 = dfDownwinnum.drop(dfDownwinnum.iloc[:,41:61],axis=1)
dfdropdown40


# In[5]:


dfdropdown40 = dfDownwinnum.drop(dfDownwinnum.iloc[:,41:61],axis=1)
dfdropdown40

dfdropup40 = dfUpwinnum.drop(dfUpwinnum.iloc[:,41:61],axis=1)
dfdropup40


# In[6]:




dfdropdown40 = dfDownwinnum.drop(dfDownwinnum.iloc[:,41:61],axis=1)
dfdropdown40

dfdropup40 = dfUpwinnum.drop(dfUpwinnum.iloc[:,41:61],axis=1)
dfdropup40


result = []
i = 0
for j in range(len(dfdropdown40["TR_id"])):
   
    
    if j == len(dfdropdown40["TR_id"])-1:
        result.append(i)
        
    elif dfdropdown40["TR_id"].iloc[j-1] != dfdropdown40["TR_id"].iloc[j]:
        result.append(i+1)
        i=i+1
          # if j ==0 append (i) 
    else:
        result.append(i)

dfdropdown40["Result"] = result  
print(dfdropdown40)


result = []
i = 0
for j in range(len(dfdropup40["TR_id"])):
   
    
    if j == len(dfdropup40["TR_id"])-1:
        result.append(i)
        
    elif dfdropup40["TR_id"].iloc[j-1] != dfdropup40["TR_id"].iloc[j]:
        result.append(i+1)
        i=i+1
          # if j ==0 append (i) 
    else:
        result.append(i)

dfdropup40["Result"] = result  
print(dfdropup40)


# In[7]:



dfdropdown40.to_csv("dfdropdown40_chrx30_5kb",index= None)
dfdropup40.to_csv("dfdropup40_chrx30_5kb",index= None)


# In[8]:




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
colsDOWN = dfdropdown40.columns.tolist()
#cols_newUP = [colsUP[-1]]
print(type(colsDOWN))
print(len(colsDOWN))
#cols_newUP.extend(colsUP[0:]) 
DownstreamMatrix=dfdropdown40
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
np.save("numpydowndrop_chrx30_5kb",DownstreamMatdropwinum)
print(DownstreamMatdropwinum)


# In[9]:



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
colsUp = dfdropup40.columns.tolist()
#cols_newUP = [colsUP[-1]]
print(type(colsUp))
print(len(colsUp))
#cols_newUP.extend(colsUP[0:]) 
UpstreamMatrix=dfdropup40
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
np.save("numpydfdropup40_chrx30_5kb",UpstreamMatdropwinum)
print(UpstreamMatdropwinum)


# In[10]:




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
colsUp = dfdropup40.columns.tolist()
#cols_newUP = [colsUP[-1]]
print(type(colsUp))
print(len(colsUp))
#cols_newUP.extend(colsUP[0:]) 
UpstreamMatrix=dfdropup40
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
np.save("numpydfdropup40_chrx30_5kb",UpstreamMatdropwinum)
print(UpstreamMatdropwinum)
# taking the log on both upstream and downstream 

log10downstreamdropwin = np.log10(DownstreamMatdropwinum)
print(log10downstreamdropwin)
log10upstreamdropwin = np.log10(UpstreamMatdropwinum)
print(log10upstreamdropwin)


# # PCA

# In[11]:


from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Downstreamdropwin= pd.DataFrame(log10downstreamdropwin)
print(Downstreamdropwin)


Downstreamdropwin.to_csv("Downstreamdrop40_chrx30_5kb", index= None)
Downstreamdropwin = pd.read_csv("Downstreamdrop40_chrx30_5kb")

Downstreamdropwin
x = Downstreamdropwin.values
y = Downstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfdownwinnum = PCA(n_components=8)
pcadfdownwinnummatrix = pcadfdownwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfdownwinnum.explained_variance_ratio_)
plt.savefig('pcadownstrea40bar_chrx30_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfdownwinnum.explained_variance_ratio_)


principaldownstreamwinnum = pd.DataFrame (data = pcadfdownwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principaldownstreamwinnum['a'], principaldownstreamwinnum['b'], c='green')

plt.savefig('pcadownstream40_chrx30_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()


# In[12]:



Upstreamdropwin= pd.DataFrame(log10upstreamdropwin)
print(Upstreamdropwin)

Upstreamdropwin.to_csv("Upstreamdrop40_chrx30_5kb.csv", index= None)

Upstreamdropwin = pd.read_csv("Upstreamdrop40_chrx30_5kb.csv")

Upstreamdropwin
x = Upstreamdropwin.values
y = Upstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfupwinnum = PCA(n_components=8)
pcadfupwinnummatrix = pcadfupwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfupwinnum.explained_variance_ratio_)
plt.savefig('pcaupstream40bar_chrx30_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfupwinnum.explained_variance_ratio_)

principalupstreamwinnum = pd.DataFrame (data = pcadfupwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalupstreamwinnum['a'], principalupstreamwinnum['b'], c='green')
plt.savefig('pcaupstream40_chrx30_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()


# In[13]:



principalupstreamwinnum.to_csv("principalupstreamwinnum40_chrx30_5kb", index= None)
principaldownstreamwinnum.to_csv("principaldownstreamwinnum40_chrx30_5kb", index= None)


# # TSNE

# // downstream

# In[14]:




from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Downstreamdropwin= pd.DataFrame(log10downstreamdropwin)
print(Downstreamdropwin)


Downstreamdropwin.to_csv("Downstreamdrop40_chrx30_5kb", index= None)
Downstreamdropwin = pd.read_csv("Downstreamdrop40_chrx30_5kb")

Downstreamdropwin
x = Downstreamdropwin.values
y = Downstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfdownwinnum = PCA(n_components=8)
pcadfdownwinnummatrix = pcadfdownwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfdownwinnum.explained_variance_ratio_)
plt.savefig('pcadownstrea40bar_chrx30_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfdownwinnum.explained_variance_ratio_)


principaldownstreamwinnum = pd.DataFrame (data = pcadfdownwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principaldownstreamwinnum['a'], principaldownstreamwinnum['b'], c='green')

plt.savefig('pcadownstream40_chrx30_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()

Upstreamdropwin= pd.DataFrame(log10upstreamdropwin)
print(Upstreamdropwin)

Upstreamdropwin.to_csv("Upstreamdrop40_chrx30_5kb.csv", index= None)

Upstreamdropwin = pd.read_csv("Upstreamdrop40_chrx30_5kb.csv")

Upstreamdropwin
x = Upstreamdropwin.values
y = Upstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfupwinnum = PCA(n_components=8)
pcadfupwinnummatrix = pcadfupwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfupwinnum.explained_variance_ratio_)
plt.savefig('pcaupstream40bar_chrx30_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfupwinnum.explained_variance_ratio_)

principalupstreamwinnum = pd.DataFrame (data = pcadfupwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalupstreamwinnum['a'], principalupstreamwinnum['b'], c='green')
plt.savefig('pcaupstream40_chrx30_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()

principalupstreamwinnum.to_csv("principalupstreamwinnum40_chrx30_5kb", index= None)
principaldownstreamwinnum.to_csv("principaldownstreamwinnum40_chrx30_5kb", index= None)

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
plt.savefig('tsnedownstream,perplexity:1000_chrx30_5kb(40).pdf', dpi=299)

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
plt.savefig('tsnedownstream,perplexity:50_chrx30_5kb(40).pdf', dpi=299)
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
plt.savefig('tsnedownstream,perplexity:5_chrx30_5kb(40).pdf', dpi=299)
plt.show()


# // upstream

# In[15]:



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
X_2u1 = tsne.fit_transform(Z)



X_2u1 = pd.DataFrame (data = X_2u1, columns = ['a', 'b'])
plt.scatter(X_2u1 ['a'], X_2u1 ['b'], c='green')
plt.title('With perplexity = 1000, tsne for upstream Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:1000_chrx30_5kb(40).pdf', dpi=299)
plt.show()


n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=50,
              random_state=12)
Z= Upstreamdropwin.values
X_2u2 = tsne.fit_transform(Z)



X_2u2 = pd.DataFrame (data = X_2u2, columns = ['a', 'b'])
plt.scatter(X_2u2 ['a'], X_2u2 ['b'], c='green')

plt.title('With perplexity = 50, tsne for Upstreamm Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:50_chrx30_5kb(40).pdf', dpi=299)
plt.show()

n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=5,
              random_state=12)
Z= Upstreamdropwin.values
X_2u = tsne.fit_transform(Z)



X_2u = pd.DataFrame (data = X_2u, columns = ['a', 'b'])
plt.scatter(X_2u ['a'], X_2u ['b'], c='green')
plt.title('With perplexity = 5, tsne for Upstream Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:5_chrx30_5kb(40).pdf', dpi=299)
plt.show()


# In[16]:


X_2u2.to_csv("X_2u240_chrx30_5kb.csv", index= None)
X_2u1.to_csv("X_2u140_chrx30_5kb.csv", index= None)
X_2u.to_csv("X_2u40_chrx30_5kb.csv", index= None)
X_2d.to_csv("X_2d40_chrx30_5kb.csv", index= None)
X_2d1.to_csv("X_2d140_chrx30_5kb.csv", index= None)
X_2d2.to_csv("X_2d240_chrx30_5kb.csv", index= None)


# # Using the Dictionary, Plot the seqclass on Reduced Dimentionality Cluster

# In[17]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfdropdown40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRDOWN_40 =dfdropdown40[str(9+seqclass)].sum()
    
    return sumTRDOWN_40

colsDOWN = list()
colsDOWN = dfdropdown40.columns.tolist()
print(type(colsDOWN))
print(len(colsDOWN))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRDOWN_40= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfdropdown40["Result"].tolist())):
    for key in translater.keys():
        sumTRDOWN_40[key].append(sum_df(dfdropdown40,i, translater[key]))

dfDownwinnumdrop40 = dfdropdown40


# In[18]:



def sum_df(df, tr_id, seqclass):
    #print (df)
    dfdropup40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRUP_40 =dfdropup40[str(9+seqclass)].sum()
    
    return sumTRUP_40

colsUP = list()
colsUP = dfdropup40.columns.tolist()
print(type(colsUP))
print(len(colsUP))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRUP_40= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfdropup40["Result"].tolist())):
    for key in translater.keys():
        sumTRUP_40[key].append(sum_df(dfdropup40,i, translater[key]))

dfUpwinnumdrop40 = dfdropup40


# In[19]:


from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# // upstream

# In[20]:



# histogram
for key in translater.keys():
    plt.hist(sumTRUP_40[key])
    plt.title("Bar, seqClass: "+ key + ", Upstream data")
    plt.savefig("seqClass" + key+ "upstream40_chrx30_5kb.pdf",dpi=299)
    plt.show()
    
for key in translater.keys():
    plt.plot(sumTRUP_40[key])
    plt.title(key)
    plt.savefig('upstream sumTR seqClass line', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()
    
# chrx30_5kb_up_tsne, perplex: 5   
for key in translater.keys():

    X_2u1[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2u1, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Upstream data")
    plt.savefig("seqClass" + key+ "upstream40_tsne:5_chrx30_5kb.pdf",dpi=299)
    plt.show()
    
# chrx30_5kb_up_tsne, perplex: 50    
for key in translater.keys():

    X_2u2[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2u2, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 50, seqClass: "+ key + ", Upstream data")
    plt.savefig("seqClass" + key+ "upstream40_tsne:50_chrx30_5kb.pdf",dpi=299)
    plt.show()

# chrx30_5kb_up_pca
for key in translater.keys():

    principalupstreamwinnum[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue=key)
    plt.title("pca, seqClass: "+ key + ", Upstream data")
    plt.savefig("seqClass" + key+ "upstream40_pca_chrx30_5kb.pdf",dpi=299)
    plt.show()


# // downstream

# In[21]:


# histogram
for key in translater.keys():
    plt.hist(sumTRDOWN_40[key])
    plt.title("Bar, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream40_chrx30_5kb.pdf",dpi=299)
    plt.show()
    
for key in translater.keys():
    plt.plot(sumTRDOWN_40[key])
    plt.title(key)
    plt.show()
    
# chrx30_5kb_down_tsne, perplex: 5
for key in translater.keys():

    X_2d[key] = sumTRDOWN_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream40_tsne:5_chrx30_5kb.pdf",dpi=299)
    plt.show()
    
# chrx30_5kb_down_tsne, perplex: 50
for key in translater.keys():

    X_2d1[key] = sumTRDOWN_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d1, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream40_tsne:50_chrx30_5kb.pdf",dpi=299)
    plt.show()
 
 # chrx30_5kb_down_pca  
for key in translater.keys():

    principaldownstreamwinnum[key] = sumTRDOWN_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue=key)
    plt.title("pca, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream40_pca_chrx30_5kb.pdf",dpi=299)
    plt.show()


# # UMAP

# In[22]:


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


# In[23]:


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


# // downstream full matrix

# In[24]:




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

    Xdowndf[key] = sumTRDOWN_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdowndf, x=Xdowndf[0],y=Xdowndf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream data-prereduction_chrx30_5kb")
    plt.savefig("seqClass" + key+ "Downstream_umap(prereduct)(40)_chrx30_5kb.pdf",dpi=299)
    plt.show()


# In[25]:


dfdropdown40


# // downstream after subsetting the df

# In[26]:



# downstream plt sum of seq TR downstream 

def sum_df(df, tr_id, seqclass):
    #print (df)
    dfdropdown40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRpltdown =dfdropdown40[str(seqclass)].sum()
    return sumTRpltdown



colsDown = list()
colsDown = dfdropdown40.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)
sumTRpltdown= {}

cnt = 0

for tr_id in list(set(dfdropdown40["Result"].tolist())):
    sumTRpltdown[tr_id]=[]
    for seqclass in range(9,49):
        
        sumTRpltdown[tr_id].append(sum_df(dfdropdown40,tr_id, seqclass))
    dfDownwinnumdrop = dfdropdown40
    
for tr_id in range(9,49):
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

    Xdownumapdf[key] = sumTRDOWN_40 [key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdownumapdf, x=Xdownumapdf[0],y=Xdownumapdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream data_chrx30_5kb")
    plt.savefig("seqClass" + key+ "downstream40_umap_chrx30_5kb.pdf",dpi=299)
    plt.show()


# // upstream umap full matrix

# In[27]:


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

    XUpdf[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=XUpdf, x=XUpdf[0],y=XUpdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Upstream dataaaa-prereduction_chrx30_5kb")
    plt.savefig("seqClass" + key+ "Upstream_umap(prereduct)40_chrx30_5kb.pdf",dpi=299)
    plt.show()


# // upstream after subsetting the df

# In[28]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfdropup40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRpltup =dfdropup40[str(seqclass)].sum()
    return sumTRpltup


colsUp = list()
colsUp = dfdropup40.columns.tolist()
print(type(colsUp))
print(len(colsUp))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)
sumTRpltup = {}

cnt = 0

for tr_id in list(set(dfdropup40["Result"].tolist())):
    sumTRpltup[tr_id]=[]
    for seqclass in range(9,49):
        
        sumTRpltup[tr_id].append(sum_df(dfdropup40,tr_id, seqclass))
    dfUpwinnumdrop = dfdropup40
    
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

    Xupumapdf[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xupumapdf, x=Xupumapdf[0],y=Xupumapdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Upstream data_chrx30_5kb")
    plt.savefig("seqClass" + key+ "Upstream40_umap_chrx30_5kb.pdf",dpi=299)
    plt.show()
    
reductupumap2d_chrx30_5kb=pd.DataFrame(Xupumap)


# In[29]:


reductdownumap2d=pd.DataFrame(Xdownumap)


# In[30]:


reductdownumap2d=pd.DataFrame(Xdownumap)

umap2dimensiondown_chrx30_5kb=pd.DataFrame(Xdown)
print(umap2dimensiondown_chrx30_5kb)
umap2dimensionup_chrx30_5kb=pd.DataFrame(XUp)
print(umap2dimensionup_chrx30_5kb)


# In[31]:


reductdownumap2d=pd.DataFrame(Xdownumap)
umap2dimensiondown_chrx30_5kb=pd.DataFrame(Xdown)
print(umap2dimensiondown_chrx30_5kb)
umap2dimensionup_chrx30_5kb=pd.DataFrame(XUp)
print(umap2dimensionup_chrx30_5kb)

# save the dfs
dfsumTRpltup.to_csv("dfsumTRpltup40_chrx30_5kb.tsv", index= None)
dfsumTRpltdown.to_csv("dfsumTRpltdown40_chrx30_5kb.tsv", index= None)
umap2dimensionup_chrx30_5kb.to_csv("umap2dimensionup40_chrx30_5kb.tsv", index= None)
umap2dimensiondown_chrx30_5kb.to_csv("umap2dimensiondown40_chrx30_5kb.tsv", index= None)
# reduced 
reductdownumap2d.to_csv("reductdownumap2d40_chrx30_5kb.tsv", index= None)
reductupumap2d_chrx30_5kb.to_csv("reductupumap2d40_chrx30_5kb.tsv", index= None)


# # Loading the saved CSVs just in case

# In[29]:


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
X_2u2=  pd.read_csv("X_2u240_chrx30_5kb.csv")
X_2u1= pd.read_csv("X_2u140_chrx30_5kb.csv")
X_2u= pd.read_csv("X_2u40_chrx30_5kb.csv")
X_2d= pd.read_csv("X_2d40_chrx30_5kb.csv")
X_2d1= pd.read_csv("X_2d140_chrx30_5kb.csv")
X_2d2= pd.read_csv("X_2d240_chrx30_5kb.csv")
# pca
principalupstreamwinnum= pd.read_csv("principalupstreamwinnum40_chrx30_5kb")
principaldownstreamwinnum= pd.read_csv("principaldownstreamwinnum40_chrx30_5kb")


#full matrix 


dfdropdown40= pd.read_csv("dfdropdown40_chrx30_5kb")
dfdropup40= pd.read_csv("dfdropup40_chrx30_5kb")

# after taking the log, pca input 
Downstreamdropwin = pd.read_csv("Downstreamdrop40_chrx30_5kb")
Upstreamdropwin = pd.read_csv("Upstreamdrop40_chrx30_5kb.csv")



# In[2]:


# save the dfs
dfsumTRpltup= pd.read_csv("dfsumTRpltup40_chrx30_5kb.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown40_chrx30_5kb.tsv")
umap2dimensionup_chrx= pd.read_csv("umap2dimensionup40_chrx30_5kb.tsv")
umap2dimensiondown_chrx= pd.read_csv("umap2dimensiondown40_chrx30_5kb.tsv")
# reduced 
reductdownumap2d= pd.read_csv("reductdownumap2d40_chrx30_5kb.tsv")
reductupumap2d_chrx30_5kb= pd.read_csv("reductupumap2d40_chrx30_5kb.tsv")


# In[3]:




#louvain
upresult= pd.read_csv("upresult40_chrx30_5kb.tsv")
downresult= pd.read_csv("downresult40_chrx30_5kb.tsv")
dfupprediction= pd.read_csv("dfupprediction40_chrx30_5kb.tsv")
dfdownprediction= pd.read_csv("dfdownprediction40_chrx30_5kb.tsv")
# louvain byw
dfdownpredictionbyw= pd.read_csv("downpredictionbyw40_chrx30_5kb.tsv")
dfuppredictionbyw= pd.read_csv("uppredictionbyw40_chrx30_5kb.tsv")


# # Louvain Clustering

# In[4]:


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


# In[5]:



n_clusters = 6
n_features=205
n_samples=20639
random_state = 42


updata, truth = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state, n_features=n_features)
updata = preprocessing.MinMaxScaler().fit_transform(Upstreamdropwin)

# Plot
plt.scatter(updata[:, 0], updata[:, 1], s=50, c = truth, cmap = 'viridis')
plt.title(f"Example of a mixture of {n_clusters} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2");

n_clusters = 6
n_features=205
n_samples=20639
random_state = 42


downdata, truth = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state, n_features=n_features)
downdata = preprocessing.MinMaxScaler().fit_transform(Downstreamdropwin)
downdata 

# Plot
plt.scatter(downdata[:, 0], downdata[:, 1], s=50, c = truth, cmap = 'viridis')
plt.title(f"Example of a mixture of {n_clusters} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2");


# In[6]:


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


# In[7]:


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


# In[8]:


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
    
downdistanceMatrix =  euclidean_distances(downdata, downdata)
print(downdistanceMatrix.shape)
updistanceMatrix =  euclidean_distances(updata, updata)
print(updistanceMatrix.shape)


# In[ ]:





# In[9]:


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


# In[10]:


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



downprediction = cluster_by_distance_matrix(downdistanceMatrix)
Counter(downprediction)

upprediction = cluster_by_distance_matrix(updistanceMatrix)
Counter(upprediction)


# In[11]:


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



downprediction = cluster_by_distance_matrix(downdistanceMatrix)
Counter(downprediction)

upprediction = cluster_by_distance_matrix(updistanceMatrix)
Counter(upprediction)


adjusted_rand_score(truth, downprediction)
adjusted_rand_score(truth, upprediction)


# In[12]:


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



downprediction = cluster_by_distance_matrix(downdistanceMatrix)
Counter(downprediction)

upprediction = cluster_by_distance_matrix(updistanceMatrix)
Counter(upprediction)


adjusted_rand_score(truth, downprediction)
adjusted_rand_score(truth, upprediction)

dfdownprediction=pd.DataFrame(downprediction)
print(dfdownprediction)

dfupprediction=pd.DataFrame(upprediction)
print(dfupprediction)


# In[13]:


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



downprediction = cluster_by_distance_matrix(downdistanceMatrix)
Counter(downprediction)

upprediction = cluster_by_distance_matrix(updistanceMatrix)
Counter(upprediction)


adjusted_rand_score(truth, downprediction)
adjusted_rand_score(truth, upprediction)

dfdownprediction=pd.DataFrame(downprediction)
print(dfdownprediction)

dfupprediction=pd.DataFrame(upprediction)
print(dfupprediction)

downresult = pd.concat([Downstreamdropwin, dfdownprediction], axis=1)
downresult.columns = [*downresult.columns[:-1], 'down_prediction']
upresult = pd.concat([Upstreamdropwin, dfupprediction], axis=1)
upresult.columns = [*upresult.columns[:-1], 'up_prediction']


# In[14]:


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



downprediction = cluster_by_distance_matrix(downdistanceMatrix)
Counter(downprediction)

upprediction = cluster_by_distance_matrix(updistanceMatrix)
Counter(upprediction)


adjusted_rand_score(truth, downprediction)
adjusted_rand_score(truth, upprediction)

dfdownprediction=pd.DataFrame(downprediction)
print(dfdownprediction)

dfupprediction=pd.DataFrame(upprediction)
print(dfupprediction)

downresult = pd.concat([Downstreamdropwin, dfdownprediction], axis=1)
downresult.columns = [*downresult.columns[:-1], 'down_prediction']
upresult = pd.concat([Upstreamdropwin, dfupprediction], axis=1)
upresult.columns = [*upresult.columns[:-1], 'up_prediction']

upresult.to_csv("upresult40_chrx30_5kb.tsv", index= None)
downresult.to_csv("downresult40_chrx30_5kb.tsv", index= None)
dfupprediction.to_csv("dfupprediction40_chrx30_5kb.tsv", index= None)
dfdownprediction.to_csv("dfdownprediction40_chrx30_5kb.tsv", index= None)
dfdownpredictionbyw=pd.DataFrame(downpredictionbyw)
dfuppredictionbyw=pd.DataFrame(uppredictionbyw)
dfdownpredictionbyw.to_csv("downpredictionbyw40_chrx30_5kb.tsv", index= None)
dfuppredictionbyw.to_csv("uppredictionbyw40_chrx30_5kb.tsv", index= None)


# In[15]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=upresult, x=upresult['0'], y=upresult['1'], hue='up_prediction')
plt.title("up louvain")
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=downresult, x=downresult['0'], y=downresult['1'], hue='down_prediction')
plt.title("down louvain")
plt.show()


# In[16]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=upresult, x=upresult['0'], y=upresult['1'], hue='up_prediction')
plt.title("up louvain")
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=downresult, x=downresult['0'], y=downresult['1'], hue='down_prediction')
plt.title("down louvain")
plt.show()

for key in resolution_parameter.keys():

    principalupstreamwinnum[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue=key, palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream data")
    plt.savefig("louvain_res_" + key+ "_pca40_up_chrx30_5kb.pdf",dpi=299)
    plt.show()


# In[17]:


for key in resolution_parameter.keys():

    principaldownstreamwinnum[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue=key, palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Downstream data")
    plt.savefig("louvain_res_" + key+ "_pca40_down_chrx30_5kb.pdf",dpi=299)
    plt.show()


# In[18]:


for key in resolution_parameter.keys():

    reductupumap2d_chrx30_5kb[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=reductupumap2d_chrx30_5kb, x='0', y='1', hue=key, palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream dataa")
    plt.savefig("louvain_res_" + key+ "_umap_sumTR40_up_chrx30_5kb.pdf",dpi=299)
    plt.show()


# In[19]:


for key in resolution_parameter.keys():

    reductdownumap2d[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue=key,palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Downstream data")
    plt.savefig("louvain_res_" + key+ "_umap_sumTR40_down_chrx30_5kb.pdf",dpi=299)
    plt.show()


# # Heatmap Analysis

# In[21]:


dfdropup40


# In[23]:


dfdropup40.drop(['TR_id', 'Win_num','Result'], axis =1)


# In[27]:


#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#upstream
pca = PCA()
df_pca = pca.fit_transform(dfdropup40)

most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

mostimportant10= most_important10_differentmethod[:,0]


up_mostimportantsubset=dfdropup40[mostimportant10]
up_mostimportantsubset

importantupresult = pd.concat([up_mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']

from sklearn.decomposition import PCA
pcadfupstreamMatreal = PCA(n_components=2)
principalComponentsdfupstreamMatreal = pcadfupstreamMatreal.fit_transform(dfdropup40)

plt.bar(x=range(2), height= pcadfupstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfupstreamMatreal.explained_variance_ratio_)

principalupstreamDfreal = pd.DataFrame (data = principalComponentsdfupstreamMatreal, columns = ['a', 'b'])
plt.scatter(principalupstreamDfreal['a'], principalupstreamDfreal['b'], c='purple')
plt.title('most important 10 for upstream full Matrix')
plt.savefig('upstream40_mostimportant10_chrx30_5kb.pdf', dpi=299)
plt.show()

#downstream


pca = PCA()
df_pca = pca.fit_transform(dfdropdown40)

down_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

down_mostimportant10= down_most_important10_differentmethod[:,0]
down_mostimportant10= list(down_mostimportant10)

down_most_important10_differentmethod
dfdropdown40.columns = dfdropdown40.columns.astype(int) 
dfdropdown40.columns 
down_mostimportantsubset=dfdropdown40[down_mostimportant10]
down_mostimportantsubset

from sklearn.decomposition import PCA
pcadfdownstreamMatreal = PCA(n_components=2)
principalComponentsdfdownstreamMatreal = pcadfdownstreamMatreal.fit_transform(dfdropdown40)

plt.bar(x=range(2), height= pcadfdownstreamMatreal.explained_variance_ratio_)

plt.show()

sum(pcadfdownstreamMatreal.explained_variance_ratio_)

principaldownstreamDfreal = pd.DataFrame (data = principalComponentsdfdownstreamMatreal, columns = ['a', 'b'])
plt.scatter(principaldownstreamDfreal['a'], principaldownstreamDfreal['b'], c='purple')
plt.title('most important 10 for downstream full Matrix')
plt.savefig('downstream40_mostimportant10_chx30_5kb.pdf', dpi=299)
plt.show()




#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(dfdropup40)

up_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

up_mostimportant10= up_most_important10_differentmethod[:,0]
up_mostimportant10= list(up_mostimportant10)

mostimportantsubset=Upstreamdropwin[up_mostimportant10]
mostimportantsubset

importantupresult = pd.concat([mostimportantsubset, dfupprediction], axis=1)
importantupresult .columns = [*importantupresult .columns[:-1], 'p']
 
#pca.component , upstream first 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA()
df_pca = pca.fit_transform(dfdropdown40)

down_most_important10_differentmethod=np.abs(pca.components_).argsort()[::-1][:10]

down_mostimportant10= down_most_important10_differentmethod[:,0]
down_mostimportant10= list(down_mostimportant10)
                          
down_most_important10_differentmethod
dfdropdown40.columns = dfdropdown40.columns.astype(int) 
dfdropdown40.columns 
down_mostimportantsubset=dfdropdown40[down_mostimportant10]
down_mostimportantsubset
importantdownresult = pd.concat([down_mostimportantsubset, dfdownprediction], axis=1)
importantdownresult.columns = [*importantdownresult .columns[:-1], 'p']

importantdownresult
importantupresult
meanupimportant = importantupresult.groupby('p').mean()
meanupdf=pd.DataFrame(meanupimportant)

print(meanupdf)
                          
                          
meandownimportant = importantdownresult.groupby('p').mean()
meandowndf=pd.DataFrame(meandownimportant)
meandowndf


countdown = importantdownresult.value_counts('p')
countup =importantupresult.value_counts('p')
dfcountup=pd.DataFrame(countup)
dfcountdown=pd.DataFrame(countdown)

dfcountupwhere = np.where((dfcountup[0]<10),False,True)
dfcountdownwhere = np.where((dfcountdown[0]<10),False,True)


# In[32]:


def sum_df_down(df, tr_id, seqclass):
    #print (df)
    dfdropdown40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRDown_40 =dfdropdown40 [str(9+seqclass)].sum()
    
    return sumTRDown_40



colsDown = list()
colsDown = dfdropdown40.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRDown_40= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfdropdown40["Result"].tolist())):
    for key in translater.keys():
        sumTRDown_40 [key].append(sum_df_down(dfdropdown40,i, translater[key]))

dfDownwinnumdrop = dfdropdown40


def sum_df_up(df, tr_id, seqclass):
    #print (df)
    dfdropup40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRUP_40 =dfdropup40[str(9+seqclass)].sum()
    
    return sumTRUP_40

colsUP = list()
colsUP = dfdropup40.columns.tolist()
print(type(colsUP))
print(len(colsUP))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRUP_40= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfdropup40["Result"].tolist())):
    for key in translater.keys():
        sumTRUP_40[key].append(sum_df_up(dfdropup40,i, translater[key]))

dfUpwinnumdrop = dfdropup40


import pandas as pd

dfsumTRpltdown = np.log10(dfsumTRpltdown)
dfsumTRpltdown
dfsumTRpltup = np.log10(dfsumTRpltup)
dfsumTRpltup

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
plt.title('most important 10 for upstream 60Seqclassdf_HeatMap_chrx30_5kb')
plt.savefig('upstream40_mostimportant10_HeatMap_60seqclass_chrx30_5kb.pdf', dpi=299)
sns.clustermap(upheatsum60)

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
plt.title('most important 10 for downstream 60Seqclassdf_HeatMap_chrx30_5kb')
plt.savefig('downstream40_mostimportant10_HeatMap_60seqclass_chrx30_5kb.pdf', dpi=299)
sns.clustermap(downheatsum60)


# # Zscore and Heatmap

# In[33]:


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
plt.title('upstream zscore_HeatMap_chrx30_5kb')
plt.savefig('upstream40_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoreup60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreup60)

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
plt.title('downstream zscore_HeatMa_chrx30_5kb')
plt.savefig('downstream40_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60)


# # Labeling and Heatmap

# In[34]:


reductupumap2d_chrx30_5kb['p'] = dfuppredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductupumap2d_chrx30_5kb, x=reductupumap2d_chrx30_5kb['0'], y=reductupumap2d_chrx30_5kb['1'], hue='p',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfdownpredictionbyw
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p',palette="tab10")
plt.show()


dfUpseqname = dfdropup40.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6'})
dfUpseqname
dfUpseqname=dfUpseqname.drop(columns=['TR_id', 'Win_num','Result'])

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
plt.title('upstream zscore_HeatMap_chrx30_5kb')
plt.savefig('upstream40_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredup60seq)

dfDownseqname = dfdropdown40.rename(columns={'8': 'PC1', '9': 'L1', '10': 'TN1', '11': 'TN2', '12': 'L2', '13':'E1', '14':'E2','15': 'E3', '16': 'L3', '17':'E4', '18': 'TF1', '19': 'HET1', '20': 'E5', '21': 'E6', '22':'TF2', '23': 'PC2', '24': 'E7', '25': 'E8', '26': 'L4', '27':'TF3','28':'PC3','29': 'E7','30':'TN3','31':'L5','32':'HET5','33':'L6','34':'P','35': 'E9','36':'CTCF','37':'TN4','38':'HET3','39':'E10','40':'TF4','41':'HET4','42':'L7','43':'PC4','44': 'HET5','45':'E11','46':'TF5', '47':'E12','48':'HET6'})
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
plt.title('downstream zscore_HeatMap_chrx30_5kb')
plt.savefig('downstream40_zscore__heatmap_chrx30_5kb.pdf', dpi=299)
sns.heatmap(meanimportantzscoreddown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoreddown60seq)


# # Labeling, Heatmap, and filteration by CTCF and Promoter 

# In[37]:


dfsumTRpltup= pd.read_csv("dfsumTRpltup_chrx30_5kb.tsv")
dfsumTRpltdown= pd.read_csv("dfsumTRpltdown_chrx30_5kb.tsv")

dfUpseqname = dfsumTRpltup.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6'})
dfUpseqname

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
plt.title('upstream zscore_HeatMap_chrx')
plt.savefig('upstream40_zscore__heatmap_chrx.pdf', dpi=299)
sns.heatmap(meanimportantzscoredup60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.set(rc={'figure.figsize':(15.7,8.27)})
sns.clustermap(meanimportantzscoredup60seq, square=False,figsize=(15.7,8.27), xticklabels=1)



dfDownseqname = dfsumTRpltdown.rename(columns={'0': 'PC1', '1': 'L1', '2': 'TN1', '3': 'TN2', '4': 'L2', '5':'E1', '6':'E2','7': 'E3', '8': 'L3', '9':'E4', '10': 'TF1', '11': 'HET1', '12': 'E5', '13': 'E6', '14':'TF2', '15': 'PC2', '16': 'E7', '17': 'E8', '18': 'L4', '19':'TF3','20':'PC3','21': 'E7','22':'TN3','23':'L5','24':'HET5','25':'L6','26':'P','27': 'E9','28':'CTCF','29':'TN4','30':'HET3','31':'E10','32':'TF4','33':'HET4','34':'L7','35':'PC4','36': 'HET5','37':'E11','38':'TF5', '39':'E12','40':'HET6'})
dfDownseqname

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
plt.title('downstream zscore_HeatMap_chrx')
plt.savefig('downstream40_zscore__heatmap_chrx.pdf', dpi=299)
sns.heatmap(meanimportantzscoredown60seq,cmap ='RdYlGn', linewidths = 0.30, annot = True)
sns.clustermap(meanimportantzscoredown60seq, square=False,figsize=(15.7,8.27), xticklabels=1)

reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()

reductupumap2d_chrx30_5kb['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chrx30_5kb['CTCF_high'] = np.where(reductupumap2d_chrx30_5kb['CTCF']>10,True,False)
sns.scatterplot(data=reductupumap2d_chrx30_5kb, x=reductupumap2d_chrx30_5kb['0'], y=reductupumap2d_chrx30_5kb['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>10,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


reductdownumap2d['CTCF'] = dfDownseqname['CTCF']
reductdownumap2d['CTCF_high'] = np.where(reductdownumap2d['CTCF']>6,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['p'] = dfDownseqname['P']
reductdownumap2d['p_high'] = np.where(reductdownumap2d['p']>9,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='p_high',palette="tab10")
plt.show()

reductupumap2d_chrx30_5kb['CTCF'] = dfUpseqname['CTCF']
reductupumap2d_chrx30_5kb['CTCF_high'] = np.where(reductupumap2d_chrx30_5kb['CTCF']>6,True,False)
sns.scatterplot(data=reductupumap2d_chrx30_5kb, x=reductupumap2d_chrx30_5kb['0'], y=reductupumap2d_chrx30_5kb['1'], hue='CTCF_high',palette="tab10")
plt.show()

reductdownumap2d['P'] = dfDownseqname['P']
reductdownumap2d['P_high'] = np.where(reductdownumap2d['P']>9,True,False)
sns.scatterplot(data=reductdownumap2d, x=reductdownumap2d['0'], y=reductdownumap2d['1'], hue='P_high',palette="tab10")
plt.show()


# # Dropping Rows based upon Mean, Median, and Max

# # mean

# In[ ]:



# upstream
meanupstream=Upstreamdropwin.mean(axis=1)
meanupstream

meandownstream = Downstreamdropwin.mean(axis = 1)
meandownstream


meandowndf =pd.concat([Downstreamdropwin,meandownstream], axis=1, join = 'inner')
meandowndf.columns = [*meandowndf.columns[:-1], 'mean']
meandowndf

meanupdf =pd.concat([Upstreamdropwin,meanupstream], axis=1, join = 'inner')
meanupdf.columns = [*meanupdf.columns[:-1], 'mean']
meanupdf

reductdownumap2d_chrx30_5kb['mean'] = meandowndf['mean']
plt.hist(reductdownumap2d_chrx30_5kb['mean'])

reductupumap2d['mean'] = meanupdf['mean']
plt.hist(reductupumap2d['mean'])

meandfup1 = meanupdf[meanupdf['mean'] > -1]  
meandfup1

meandfdown1 = meandowndf[meandowndf['mean'] > -1]  
meandfdown1

dfsumTRUP=pd.DataFrame(sumTRUP_40)
dfsumTRDOWN=pd.DataFrame(sumTRDOWN_40)

sumdfdown1 = dfsumTRDOWN[meandowndf['mean'] > -1]  
sumdfdown1

sumdfup1 = dfsumTRUP[meanupdf['mean'] > -1] 
sumdfup1

meandfdown1.to_csv("meandfdown140_chrx30_5kb.tsv")
meandfup1.to_csv("meandfup140_chrx30_5kb.tsv")

meandfdown1 = meandfdown1.drop(['mean'],axis =1)
meandfup1 = meandfup1.drop(['mean'],axis =1)
# downstream 
x = meandfdown1.values
y = meandfdown1.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfdowndrop = PCA(n_components=8)
pcadfdowndropmat = pcadfdowndrop.fit_transform(x)

plt.bar(x=range(8), height= pcadfdowndrop.explained_variance_ratio_)
plt.show()

sum(pcadfdowndrop.explained_variance_ratio_)


principaldownmat = pd.DataFrame (data = pcadfdowndropmat, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principaldownmat['a'], principaldownmat['b'], c='green')
plt.show()


#upstream 

x = meandfup1.values
y = meandfup1.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfupdrop = PCA(n_components=8)
pcadfupdropmat = pcadfupdrop.fit_transform(x)

plt.bar(x=range(8), height= pcadfupdrop.explained_variance_ratio_)
plt.show()

sum(pcadfupdrop.explained_variance_ratio_)


principalupmat = pd.DataFrame (data = pcadfupdropmat, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalupmat['a'], principalupmat['b'], c='green')
plt.show()


principalupmat.to_csv("principalupmat40_chrx30_5kb", index= None)
principaldownmat.to_csv("principaldownmat40_chrx30_5kb", index= None)


# # median

# # max

# # Validation

# In[ ]:


import numpy as np
import pandas as pd
ch38 = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/chrXCh38/closestsortedchrXGRCh38.bed", sep="\t", header=None, names = ['chrX','1','2','chrX1','3','4','class','distance'])

import numpy as np
import pandas as pd
ch38_chrx30 = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_downstreamAnalysis/ch38chrx_DDist.bed", sep="\t", header=None, names = ['chrX','1','2','3','chrX1','3','4','class','distance'])
ch38_chrx30

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
plt.savefig("Downstreampcawithvalidationdatadistance<5000_chrx30_5kb.pdf",dpi=299)
plt.show()


# 1000
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<1000)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 1000')
plt.savefig("Downstreampcawithvalidationdatadistance<1000_chrx30_5kb.pdf",dpi=299)
plt.show()

# 500
principaldownstreamwinnum['distance'] = np.where((ch38drop['distance']<500)&(ch38drop['distance']>=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Downstream pca with ch38drop distance of smaller than 500')
plt.savefig("Downstreampcawithvalidationdatadistance<500_chrx30_5kb.pdf",dpi=299)
plt.show()

X_2u['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='distance', palette="tab10")
plt.title( 'Upstream TSNE with ch38drop distance of larger than -2000')
plt.savefig("Upstream TSNE withvalidationdatadistance>-2000_chrx30_5kb.pdf",dpi=299)
plt.show()

X_2u1['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u1, x="a", y="b", hue='distance', palette="tab10")
plt.title('Upstream TSNE:50 with ch38drop distance of larger than -2000')
plt.savefig("Upstream TSNE:50 withvalidationdatadistance>-2000_chrx30_5kb.pdf",dpi=299)
plt.show()

X_2u['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=X_2u, x="a", y="b", hue='distance', palette="tab10")
plt.title('Upstream TSNE with ch38drop distance of larger than -2000')
plt.savefig("Upstream TSNE withvalidationdatadistance>-2000_chrx30_5kb.pdf",dpi=299)
plt.show()

principalupstreamwinnum['distance'] = np.where((ch38drop['distance']>-2000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Upstream pca with ch38drop distance of larger than -2000')
plt.savefig("Upstreampcawithvalidationdatadistance>-2000_chrx30_5kb.pdf",dpi=299)
plt.show() 

principalupstreamwinnum['distance'] = np.where((ch38drop['distance']>-1000)&(ch38drop['distance']<=0), True, False)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue='distance', palette="tab10")
plt.title('Upstream pca with ch38drop distance of larger than -1000')
plt.savefig("Upstreampcawithvalidationdatadistance>-1000_chrx30_5kb.pdf",dpi=299)
plt.show()


# # chr9_30 5kb

# In[1]:




import pandas as pd
import numpy as np


df = pd.read_csv("/data/projects/nanopore/RepeatExpansion/TR_subtyping/Sei/chr930/chromatin-profiles-hdf5/chr94colnoN_30_row_labels.txt", sep="\t",low_memory=False)
df


dfSei = np.load("/data/projects/nanopore/RepeatExpansion/TR_subtyping/TR_downstreamAnalysis/chr930/chr94colnoN_30.ref.raw_sequence_class_scores.npy")
dfSei = pd.DataFrame(dfSei)
dfSei


# concat axis default =0
dfinput = pd.concat([df,dfSei], axis = 1)
display(dfinput)


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


# In[2]:


# downstream 
dfDownstreamdropwinnum=(dfDownstreamdropwinnum.reset_index(drop=True))
dfDownwinnum = dfDownstreamdropwinnum.drop(columns=['chromosome','1','window','basepair','4','5','6','7','6','7'])
print(dfDownwinnum)
# upstream
dfUpstreamdropwinnum=(dfUpstreamdropwinnum.reset_index(drop=True))
dfUpwinnum = dfUpstreamdropwinnum.drop(columns=['chromosome','1','window','basepair','4','5','6','7','6','7'])
print(dfUpwinnum)


# In[3]:




dfdropdown40 = dfDownwinnum.drop(dfDownwinnum.iloc[:,41:61],axis=1)
dfdropdown40

dfdropup40 = dfUpwinnum.drop(dfUpwinnum.iloc[:,41:61],axis=1)
dfdropup40


result = []
i = 0
for j in range(len(dfdropdown40["TR_id"])):
   
    
    if j == len(dfdropdown40["TR_id"])-1:
        result.append(i)
        
    elif dfdropdown40["TR_id"].iloc[j-1] != dfdropdown40["TR_id"].iloc[j]:
        result.append(i+1)
        i=i+1
          # if j ==0 append (i) 
    else:
        result.append(i)

dfdropdown40["Result"] = result  
print(dfdropdown40)


result = []
i = 0
for j in range(len(dfdropup40["TR_id"])):
   
    
    if j == len(dfdropup40["TR_id"])-1:
        result.append(i)
        
    elif dfdropup40["TR_id"].iloc[j-1] != dfdropup40["TR_id"].iloc[j]:
        result.append(i+1)
        i=i+1
          # if j ==0 append (i) 
    else:
        result.append(i)

dfdropup40["Result"] = result  
print(dfdropup40)


# In[4]:




dfdropdown40.to_csv("dfdropdown40_chr930_5kb",index= None)
dfdropup40.to_csv("dfdropup40_chr930_5kb",index= None)

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
colsDOWN = dfdropdown40.columns.tolist()
#cols_newUP = [colsUP[-1]]
print(type(colsDOWN))
print(len(colsDOWN))
#cols_newUP.extend(colsUP[0:]) 
DownstreamMatrix=dfdropdown40
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
np.save("numpydowndrop_chr930_5kb",DownstreamMatdropwinum)
print(DownstreamMatdropwinum)
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
colsUp = dfdropup40.columns.tolist()
#cols_newUP = [colsUP[-1]]
print(type(colsUp))
print(len(colsUp))
#cols_newUP.extend(colsUP[0:]) 
UpstreamMatrix=dfdropup40
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
np.save("numpydfdropup40_chr930_5kb",UpstreamMatdropwinum)
print(UpstreamMatdropwinum)
# taking the log on both upstream and downstream 

log10downstreamdropwin = np.log10(DownstreamMatdropwinum)
print(log10downstreamdropwin)
log10upstreamdropwin = np.log10(UpstreamMatdropwinum)
print(log10upstreamdropwin)


# In[5]:




from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Downstreamdropwin= pd.DataFrame(log10downstreamdropwin)
print(Downstreamdropwin)


Downstreamdropwin.to_csv("Downstreamdrop40_chr930_5kb", index= None)
Downstreamdropwin = pd.read_csv("Downstreamdrop40_chr930_5kb")

Downstreamdropwin
x = Downstreamdropwin.values
y = Downstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfdownwinnum = PCA(n_components=8)
pcadfdownwinnummatrix = pcadfdownwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfdownwinnum.explained_variance_ratio_)
plt.savefig('pcadownstrea40bar_chr930_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfdownwinnum.explained_variance_ratio_)


principaldownstreamwinnum = pd.DataFrame (data = pcadfdownwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principaldownstreamwinnum['a'], principaldownstreamwinnum['b'], c='green')

plt.savefig('pcadownstream40_chr930_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()

Upstreamdropwin= pd.DataFrame(log10upstreamdropwin)
print(Upstreamdropwin)

Upstreamdropwin.to_csv("Upstreamdrop40_chr930_5kb.csv", index= None)

Upstreamdropwin = pd.read_csv("Upstreamdrop40_chr930_5kb.csv")

Upstreamdropwin
x = Upstreamdropwin.values
y = Upstreamdropwin.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pcadfupwinnum = PCA(n_components=8)
pcadfupwinnummatrix = pcadfupwinnum.fit_transform(x)

plt.bar(x=range(8), height= pcadfupwinnum.explained_variance_ratio_)
plt.savefig('pcaupstream40bar_chr930_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

plt.show()

sum(pcadfupwinnum.explained_variance_ratio_)

principalupstreamwinnum = pd.DataFrame (data = pcadfupwinnummatrix, columns = ['a', 'b','c','d','e','f','g','h'])
plt.scatter(principalupstreamwinnum['a'], principalupstreamwinnum['b'], c='green')
plt.savefig('pcaupstream40_chr930_5kb.pdf', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()

principalupstreamwinnum.to_csv("principalupstreamwinnum40_chr930_5kb", index= None)
principaldownstreamwinnum.to_csv("principaldownstreamwinnum40_chr930_5kb", index= None)

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
plt.savefig('tsnedownstream,perplexity:1000_chr930_5kb(40).pdf', dpi=299)

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
plt.savefig('tsnedownstream,perplexity:50_chr930_5kb(40).pdf', dpi=299)
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
plt.savefig('tsnedownstream,perplexity:5_chrx30_5kb(40).pdf', dpi=299)
plt.show()


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
X_2u1 = tsne.fit_transform(Z)



X_2u1 = pd.DataFrame (data = X_2u1, columns = ['a', 'b'])
plt.scatter(X_2u1 ['a'], X_2u1 ['b'], c='green')
plt.title('With perplexity = 1000, tsne for upstream Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:1000_chr930_5kb(40).pdf', dpi=299)
plt.show()


n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=50,
              random_state=12)
Z= Upstreamdropwin.values
X_2u2 = tsne.fit_transform(Z)



X_2u2 = pd.DataFrame (data = X_2u2, columns = ['a', 'b'])
plt.scatter(X_2u2 ['a'], X_2u2 ['b'], c='green')

plt.title('With perplexity = 50, tsne for Upstreamm Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:50_chrx30_5kb(40).pdf', dpi=299)
plt.show()

n_components=2
tsne = TSNE(n_components=n_components,
              perplexity=5,
              random_state=12)
Z= Upstreamdropwin.values
X_2u = tsne.fit_transform(Z)



X_2u = pd.DataFrame (data = X_2u, columns = ['a', 'b'])
plt.scatter(X_2u ['a'], X_2u ['b'], c='green')
plt.title('With perplexity = 5, tsne for Upstream Matrix after dropping winnum')
plt.savefig('tsneupstream,perplexity:5_chr930_5kb(40).pdf', dpi=299)
plt.show()


# In[6]:


def sum_df(df, tr_id, seqclass):
    #print (df)
    dfdropdown40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRDOWN_40 =dfdropdown40[str(9+seqclass)].sum()
    
    return sumTRDOWN_40

colsDOWN = list()
colsDOWN = dfdropdown40.columns.tolist()
print(type(colsDOWN))
print(len(colsDOWN))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRDOWN_40= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfdropdown40["Result"].tolist())):
    for key in translater.keys():
        sumTRDOWN_40[key].append(sum_df(dfdropdown40,i, translater[key]))

dfDownwinnumdrop40 = dfdropdown40

def sum_df(df, tr_id, seqclass):
    #print (df)
    dfdropup40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRUP_40 =dfdropup40[str(9+seqclass)].sum()
    
    return sumTRUP_40

colsUP = list()
colsUP = dfdropup40.columns.tolist()
print(type(colsUP))
print(len(colsUP))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)


cnt = 0
sumTRUP_40= {'PC1': [], 'E3': [], 'E4': [], 'HET1': [], 'E8': [], 'HET2': [], 'E9': [], 'HET3':[], 'PC4' : [], 'P': [], 'CTCF' : [], 'E10' : [], 'HET4': []}
translater = {'PC1': 0, 'E3': 7, 'E4': 9, 'HET1': 11, 'E8': 17, 'HET2': 23,'E9': 26, 'HET3':29, 'PC4' :34, 'P': 25, 'CTCF' : 27, 'E10' : 30, 'HET4': 32}

for i in list(set(dfdropup40["Result"].tolist())):
    for key in translater.keys():
        sumTRUP_40[key].append(sum_df(dfdropup40,i, translater[key]))

dfUpwinnumdrop40 = dfdropup40


# In[7]:


from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# histogram
for key in translater.keys():
    plt.hist(sumTRUP_40[key])
    plt.title("Bar, seqClass: "+ key + ", Upstream data")
    plt.savefig("seqClass" + key+ "upstream40_chr930_5kb.pdf",dpi=299)
    plt.show()
    
for key in translater.keys():
    plt.plot(sumTRUP_40[key])
    plt.title(key)
    plt.savefig('upstream sumTR seqClass line', dpi=299, format='pdf', metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()
    
# chrx30_5kb_up_tsne, perplex: 5   
for key in translater.keys():

    X_2u1[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2u1, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Upstream data")
    plt.savefig("seqClass" + key+ "upstream40_tsne:5_chr930_5kb.pdf",dpi=299)
    plt.show()
    
# chrx30_5kb_up_tsne, perplex: 50    
for key in translater.keys():

    X_2u2[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2u2, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 50, seqClass: "+ key + ", Upstream data")
    plt.savefig("seqClass" + key+ "upstream40_tsne:50_chr930_5kb.pdf",dpi=299)
    plt.show()

# chrx30_5kb_up_pca
for key in translater.keys():

    principalupstreamwinnum[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue=key)
    plt.title("pca, seqClass: "+ key + ", Upstream data")
    plt.savefig("seqClass" + key+ "upstream40_pca_chr930_5kb.pdf",dpi=299)
    plt.show()
    
# histogram
for key in translater.keys():
    plt.hist(sumTRDOWN_40[key])
    plt.title("Bar, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream40_chr930_5kb.pdf",dpi=299)
    plt.show()
    
for key in translater.keys():
    plt.plot(sumTRDOWN_40[key])
    plt.title(key)
    plt.show()
    
# chrx30_5kb_down_tsne, perplex: 5
for key in translater.keys():

    X_2d[key] = sumTRDOWN_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream40_tsne:5_chr930_5kb.pdf",dpi=299)
    plt.show()
    
# chrx30_5kb_down_tsne, perplex: 50
for key in translater.keys():

    X_2d1[key] = sumTRDOWN_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=X_2d1, x="a", y="b", hue=key)
    plt.title("tsne,perplexity: 5, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream40_tsne:50_chr930_5kb.pdf",dpi=299)
    plt.show()
 
 # chrx30_5kb_down_pca  
for key in translater.keys():

    principaldownstreamwinnum[key] = sumTRDOWN_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principaldownstreamwinnum, x="a", y="b", hue=key)
    plt.title("pca, seqClass: "+ key + ", Downstream data")
    plt.savefig("seqClass" + key+ "downstream40_pca_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[8]:


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

    Xdowndf[key] = sumTRDOWN_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdowndf, x=Xdowndf[0],y=Xdowndf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream data-prereduction_chr930_5kb")
    plt.savefig("seqClass" + key+ "Downstream_umap(prereduct)(40)_chr930_5kb.pdf",dpi=299)
    plt.show()
    
# downstream plt sum of seq TR downstream 

def sum_df(df, tr_id, seqclass):
    #print (df)
    dfdropdown40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRpltdown =dfdropdown40[str(seqclass)].sum()
    return sumTRpltdown



colsDown = list()
colsDown = dfdropdown40.columns.tolist()
print(type(colsDown))
print(len(colsDown))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)
sumTRpltdown= {}

cnt = 0

for tr_id in list(set(dfdropdown40["Result"].tolist())):
    sumTRpltdown[tr_id]=[]
    for seqclass in range(9,49):
        
        sumTRpltdown[tr_id].append(sum_df(dfdropdown40,tr_id, seqclass))
    dfDownwinnumdrop = dfdropdown40
    
for tr_id in range(9,49):
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

    Xdownumapdf[key] = sumTRDOWN_40 [key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xdownumapdf, x=Xdownumapdf[0],y=Xdownumapdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Downstream data_chr930_5kb")
    plt.savefig("seqClass" + key+ "downstream40_umap_chr930_5kb.pdf",dpi=299)
    plt.show()


# In[9]:


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

    XUpdf[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=XUpdf, x=XUpdf[0],y=XUpdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Upstream data-prereduction_chr930_5kb")
    plt.savefig("seqClass" + key+ "Upstream_umap(prereduct)40_chr930_5kb.pdf",dpi=299)
    plt.show()
    
def sum_df(df, tr_id, seqclass):
    #print (df)
    dfdropup40 = df.loc[df["Result"] == (tr_id)].copy()
    sumTRpltup =dfdropup40[str(seqclass)].sum()
    return sumTRpltup


colsUp = list()
colsUp = dfdropup40.columns.tolist()
print(type(colsUp))
print(len(colsUp))
#cols_newDOWN.extend(colsDOWN[0:]) 

#print(DownstreamMatrix.columns)
sumTRpltup = {}

cnt = 0

for tr_id in list(set(dfdropup40["Result"].tolist())):
    sumTRpltup[tr_id]=[]
    for seqclass in range(9,49):
        
        sumTRpltup[tr_id].append(sum_df(dfdropup40,tr_id, seqclass))
    dfUpwinnumdrop = dfdropup40
    
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

    Xupumapdf[key] = sumTRUP_40[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=Xupumapdf, x=Xupumapdf[0],y=Xupumapdf[1], hue=key)
    plt.title("umap, seqClass: "+ key + ", Upstream data_chr930_5kb")
    plt.savefig("seqClass" + key+ "Upstream40_umap_chr930_5kb.pdf",dpi=299)
    plt.show()
    
reductupumap2d_chr930_5kb=pd.DataFrame(Xupumap)


# In[10]:


reductdownumap2d=pd.DataFrame(Xdownumap)
umap2dimensiondown_chrx30_5kb=pd.DataFrame(Xdown)
print(umap2dimensiondown_chrx30_5kb)
umap2dimensionup_chrx30_5kb=pd.DataFrame(XUp)
print(umap2dimensionup_chrx30_5kb)

# save the dfs
dfsumTRpltup.to_csv("dfsumTRpltup40_chr930_5kb.tsv", index= None)
dfsumTRpltdown.to_csv("dfsumTRpltdown40_chr930_5kb.tsv", index= None)
umap2dimensionup_chrx30_5kb.to_csv("umap2dimensionup40_chr930_5kb.tsv", index= None)
umap2dimensiondown_chrx30_5kb.to_csv("umap2dimensiondown40_chr930_5kb.tsv", index= None)
# reduced 
reductdownumap2d.to_csv("reductdownumap2d40_chr930_5kb.tsv", index= None)
reductupumap2d_chrx30_5kb.to_csv("reductupumap2d40_chr930_5kb.tsv", index= None)

reductdownumap2d=pd.DataFrame(Xdownumap)
umap2dimensiondown_chrx30_5kb=pd.DataFrame(Xdown)
print(umap2dimensiondown_chrx30_5kb)
umap2dimensionup_chrx30_5kb=pd.DataFrame(XUp)
print(umap2dimensionup_chrx30_5kb)

# save the dfs
dfsumTRpltup.to_csv("dfsumTRpltup40_chr930_5kb.tsv", index= None)
dfsumTRpltdown.to_csv("dfsumTRpltdown40_chr930_5kb.tsv", index= None)
umap2dimensionup_chrx30_5kb.to_csv("umap2dimensionup40_chr930_5kb.tsv", index= None)
umap2dimensiondown_chrx30_5kb.to_csv("umap2dimensiondown40_chr930_5kb.tsv", index= None)
# reduced 
reductdownumap2d.to_csv("reductdownumap2d40_chr930_5kb.tsv", index= None)
reductupumap2d_chrx30_5kb.to_csv("reductupumap2d40_chr930_5kb.tsv", index= None)


# In[ ]:


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


n_clusters = 6
n_features=205
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
n_features=205
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


# In[ ]:


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



downprediction = cluster_by_distance_matrix(downdistanceMatrix)
Counter(downprediction)

upprediction = cluster_by_distance_matrix(updistanceMatrix)
Counter(upprediction)


adjusted_rand_score(truth, downprediction)
adjusted_rand_score(truth, upprediction)

dfdownprediction=pd.DataFrame(downprediction)
print(dfdownprediction)

dfupprediction=pd.DataFrame(upprediction)
print(dfupprediction)

downresult = pd.concat([Downstreamdropwin, dfdownprediction], axis=1)
downresult.columns = [*downresult.columns[:-1], 'down_prediction']
upresult = pd.concat([Upstreamdropwin, dfupprediction], axis=1)
upresult.columns = [*upresult.columns[:-1], 'up_prediction']

upresult.to_csv("upresult40_chr930_5kb.tsv", index= None)
downresult.to_csv("downresult40_chr930_5kb.tsv", index= None)
dfupprediction.to_csv("dfupprediction40_chr930_5kb.tsv", index= None)
dfdownprediction.to_csv("dfdownprediction40_chr930_5kb.tsv", index= None)
dfdownpredictionbyw=pd.DataFrame(downpredictionbyw)
dfuppredictionbyw=pd.DataFrame(uppredictionbyw)
dfdownpredictionbyw.to_csv("downpredictionbyw40_chr930_5kb.tsv", index= None)
dfuppredictionbyw.to_csv("uppredictionbyw40_chr930_5kb.tsv", index= None)


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=upresult, x=upresult['0'], y=upresult['1'], hue='up_prediction')
plt.title("up louvain")
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=downresult, x=downresult['0'], y=downresult['1'], hue='down_prediction')
plt.title("down louvain")
plt.show()

for key in resolution_parameter.keys():

    principalupstreamwinnum[key] = resolution_result[key]
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.scatterplot(data=principalupstreamwinnum, x="a", y="b", hue=key, palette="tab10")
    plt.title("Louvain Clustering, resolution: "+ key + ", Upstream data")
    plt.savefig("louvain_res_" + key+ "_pca40_up_chr930_5kb.pdf",dpi=299)
    plt.show()

