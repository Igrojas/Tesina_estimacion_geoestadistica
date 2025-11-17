#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/bd_dm_cmp_entry.csv", sep=";")
list_cols = ["midx","midy","midz", "starkey_min"]
df = df[list_cols].copy()
# %%

# Aplicar cluster k means espacial

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
coords_scaled = scaler.fit_transform(df[['midx', 'midz']])
attr_scaled = scaler.fit_transform(df[['starkey_min']])
# %%
w = 1
k = 5
coords_weight = w * coords_scaled
attr_weight = (1 - w) * attr_scaled

# Combinar
features = np.hstack([coords_weight, attr_weight])
kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(features)
df['cluster'] = clusters
#%%

sns.scatterplot(x='midx', y='midz', hue='cluster', data=df)
plt.grid(True, alpha=0.3)
plt.title('Clusters K-Means Espacial')
plt.xlabel('X')
plt.ylabel('Z')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%