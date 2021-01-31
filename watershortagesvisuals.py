import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import seaborn as sns
from pandas.plotting import scatter_matrix
from matplotlib import cm

awareness = pd.read_table('data.txt')
print('This is first')
print(awareness.head())
print(awareness.shape)
print(awareness['county'].unique())
print(awareness.groupby('county').size())
#
sns.countplot(awareness['county'], label="Count")
plt.show()
awareness.drop('county', axis=1).plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False,
                                         figsize=(9, 9),
                                         title='Box Plot for each input variable')

# plt.savefig('fruits_box')
plt.show()
#
awareness.drop('catid', axis=1).hist(bins=30, figsize=(9, 9))
pl.suptitle("Histogram for each numeric input variable")
plt.show()

awareness.drop('catid', axis=1).hist(bins=30, figsize=(9, 9))
pl.suptitle("Histogram for each numeric input variable")
plt.show()
#
feature_names = ['distance', 'dayspassed', 'categoryscore']
X = awareness[feature_names]
y = awareness['catid']
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.show()