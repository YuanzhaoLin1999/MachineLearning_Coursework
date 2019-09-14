#%%[markdown]
#Set up


#%%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')


#%%
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
%matplotlib inline          


#%%[markdown]
#1.Scatter plot


#%%
#Import dataset
midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")

#Prepare Data
#Create as many colors as there are uniquemidwest['category']
categories = np.unique(midwest['category'])
colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]
#cm代表color map，即颜色映射

#Draw Plot for Each Catefories
plt.figure(figsize=(16,10), dpi=80, facecolor='w',edgecolor='k')

for i, category in enumerate(categories):
    plt.scatter('area', 'poptotal',
    data = midwest.loc[midwest.category ==category, :],
    s = 20, color=colors[i], label=str(category))

#Decorations
plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),
              xlabel='Area', ylabel='Population')

plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)
plt.legend(fontsize=12)    
plt.show() 
#%%[markdown]
#2.Bubble plot with Encircling
#%%
from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings; warnings.simplefilter('ignore')
sns.set_style("white")

# Step 1: Prepare Data
midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")

# As many colors as there are unique midwest['category']
categories = np.unique(midwest['category'])
colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

# Step 2: Draw Scatterplot with unique color for each category
fig = plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')    

for i, category in enumerate(categories):
    plt.scatter('area', 'poptotal', 
    data=midwest.loc[midwest.category==category, :],
     s='dot_size', color=colors[i], label=str(category), 
     edgecolors='black', linewidths=.5)

#以下是特殊的部分
#Step3:Encircling
def encircle(x,y, ax=None, **kw):
    if not ax:
        ax = plt.gca() #plt.gca() 意思是取得最近的ax对象
    p = np.c_[x, y]#按行连接两个矩阵，np.r_按列连接
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)#核心是调用了这个绘图函数
    ax.add_patch(poly)

#Select data to be encircled
midwest_encircle_data = midwest.loc[midwest.state == 'IN', :]

# Draw polygon surrounding vertices    
encircle(midwest_encircle_data.area, midwest_encircle_data.poptotal, ec="k", fc="gold", alpha=0.1)
encircle(midwest_encircle_data.area, midwest_encircle_data.poptotal, ec="firebrick", fc="none", linewidth=1.5)

#%%
