import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(6, 6))

def read_at(path):
   df=pd.read_csv(path,header=2)

   #replace all instances of occurence of the double dot
   df.replace('..',0,inplace=True)

   #fil all nans with zeroes
   df=df.fillna(0)
   dft=df

   #setting country name as index in order to transpose
   #dft.set_index('Country Name',inplace=True)  

   #dft.set_index('Country Name',inplace=True)

   #transposition of df
   dftp=dft.transpose()
   return df,dft

dfframe1,dft=read_at('API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_5348180.csv')

dfframe1.head()

dfframe1['Country Name'].unique()

dfframe1.describe()

numframe1=dfframe1.drop(["Country Name"	,"Country Code","Indicator Name","Indicator Code"],axis=1).astype(float)
numframe1.head()

numframe1.describe()

x=dfframe1['Country Name'].values[180:200]
y1=numframe1["1990"].values[180:200]
y2=numframe1["2015"].values[180:200]
y3=numframe1["2021"].values[180:200]
# Plot the bars for each set of y values
plt.bar(x, y1, width=0.2, align='center', label='1990')
plt.bar([i + 0.2 for i in range(len(x))], y2, width=0.2, align='center', label='2015')
plt.bar([i + 0.4 for i in range(len(x))], y3, width=0.2, align='center', label='2021')

# Add labels and legend
plt.xlabel('countries')
plt.ylabel('Years')
plt.title('Energy use (kg of oil equivalent per capita)')
plt.xticks(rotation=90)
plt.legend()

# Show the plot
plt.show()

x = dfframe1['Country Name'].values[100:120]
y = numframe1["1990"].values[100:120]

# Create bar plot
fig, ax = plt.subplots()
ax.bar(x, y)

# Add labels and title
ax.set_xlabel('countries')
ax.set_ylabel('1990')
ax.set_title('Energy use (kg of oil equivalent per capita)')
plt.xticks(rotation=90)

# Show the plot
plt.show()

data =numframe1.iloc[10:40,10:40]

# create a heatmap using the 'hot' colormap
heatmap = plt.imshow(data, cmap='cool')

# add a colorbar to the plot
plt.colorbar(heatmap)

# set the x and y axis labels
plt.xlabel('1990-2021')
plt.ylabel('countries')
plt.xticks(np.arange(0.5, 30.5), range(1990,2020),rotation=90)
plt.yticks(np.arange(0.5, 30.5), dfframe1['Country Name'].values[10:40])

# set the plot title
plt.title('Energy use (kg of oil equivalent per capita)')

# show the plot
plt.show()

dfframe2,dft2=read_at('API_EN.ATM.CO2E.EG.ZS_DS2_en_csv_v2_4904195.csv')

dfframe2.head()

numframe2=dfframe2.drop(["Country Name"	,"Country Code","Indicator Name","Indicator Code"],axis=1).astype(float)
numframe2.head()

numframe2.describe()

x = dfframe2['Country Name'].values[10:40]
y1 = numframe2["1990"].values[10:40]
y2 = numframe2["2005"].values[10:40]

# Create multiline plot
fig, ax = plt.subplots()
ax.plot(x, y1, 'bo-', label='1990')
ax.plot(x, y2, 'rs-', label='2021')

# Add labels and legend
ax.set_xlabel('countries')
ax.set_ylabel('years')
ax.set_title('CO2 intensity (kg per kg of oil equivalent energy use)')
ax.legend()
plt.xticks(rotation=90)

# Show the plot
plt.show()

import warnings
warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")


x = dfframe2['Country Name'].values[80:100]
y1 = numframe2['1990'].values[80:100]
y2 = numframe2['1995'].values[80:100]
y3 = numframe2['2000'].values[80:100]

# Create the stacked bar chart
fig, ax = plt.subplots()
ax.bar(x, y1, label='1990')
ax.bar(x, y2, bottom=y1, label='1995')
ax.bar(x, y3, bottom=y1+y2, label='2000')
ax.set_xticklabels(x, rotation=90)

# Add legend and labels
ax.legend()
ax.set_title('CO2 intensity (kg per kg of oil equivalent energy use)')
ax.set_xlabel('countries')
ax.set_ylabel('1990,1995,2000')


# Show the plot
plt.show()

data =numframe2.iloc[30:50,30:50]

# create a heatmap using the 'hot' colormap
heatmap = plt.imshow(data, cmap='YlGn')

# add a colorbar to the plot
plt.colorbar(heatmap)

# set the x and y axis labels
plt.xlabel('1990-2010')
plt.ylabel('countries')
plt.xticks(np.arange(0.5, 20.5), range(1990,2010),rotation=90)
plt.yticks(np.arange(0.5, 20.5), dfframe2['Country Name'].values[30:50])

# set the plot title
plt.title('CO2 intensity (kg per kg of oil equivalent energy use)\n of twenty countries from 1990-2010')

# show the plot
plt.show()