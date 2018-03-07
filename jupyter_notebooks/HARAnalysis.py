
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import copy
import glob
import gzip
import itertools
import matplotlib
from matplotlib import colors
from matplotlib import pyplot
import numpy
import os
import pickle
import random
import scipy.sparse as sps
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy import stats
from sklearn.cluster import dbscan, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, calinski_harabaz_score, consensus_score
import smart_open
import sparkpickle
import struct
import sys


# In[2]:

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# ##Define Functions

# In[3]:

from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_bic(kmeans,X):
    """
    Thanks to the following forum thread for this function code: 
    https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)


# ##Load ncHAR Data
# 
# Good site for learning about and playing with t-SNE: http://distill.pub/2016/misread-tsne/

# In[4]:

'''Read in data that has been averaged over the 2640 ncHARs. We have three datasets:
1) The observed data (Roadmap Epigenomics Consolidated)
2) The imputed data (PREDICTD)
3) The observed data at randomized ncHAR positions
'''
path_to_observed_data = 's3://predictd/reference_data/ncHAR_analysis/obs_data.avg_signal_per_nchar.pickle'
with smart_open.smart_open(path_to_observed_data, 'rb') as pickle_in:
    har_info, har_data = pickle.loads(pickle_in.read())
path_to_imputed_data = 's3://predictd/reference_data/ncHAR_analysis/imp_data.avg_signal_per_nchar.pickle'
with smart_open.smart_open(path_to_imputed_data, 'rb') as pickle_in:
    har_info_imp, har_data_imp = pickle.loads(pickle_in.read())
path_to_randomly_shuffled_data = 's3://predictd/reference_data/ncHAR_analysis/obs_data.avg_signal_per_nchar.shuffled.pickle'
with smart_open.smart_open(path_to_randomly_shuffled_data, 'rb') as pickle_in:
    har_info_rand, har_data_rand = pickle.loads(pickle_in.read())
    
har_data = numpy.nan_to_num(har_data)
har_data_imp = numpy.nan_to_num(har_data_imp)
har_data_rand = numpy.nan_to_num(har_data_rand)

#ensure same sort order
har_info, sort_idx = zip(*[(elt, har_info.index(elt)) for elt in sorted(har_info, key=lambda x: (x[0], x[1]))])
har_data = har_data[sort_idx,:]

har_info_imp, sort_idx = zip(*[(elt, har_info_imp.index(elt)) for elt in sorted(har_info_imp, key=lambda x: (x[0], x[1]))])
har_data_imp = har_data_imp[sort_idx,:]

har_info_rand, sort_idx = zip(*[(elt, har_info_rand.index(elt)) for elt in sorted(har_info_rand, key=lambda x: (x[0], x[1]))])
har_data_rand = har_data_rand[sort_idx,:]


# In[5]:

print(har_data.shape)
print(har_data_imp.shape)
print(har_data_rand.shape)
print(har_info[0])
print(har_info_imp[0])
print(har_info_rand[0])


# In[6]:

'''Load in the coordinates of cell types and assays in the data structures (data_idx.pickle), 
and the locations of observed data (folds.5.pickle) so we can extract the relevant data for this
ncHAR analysis.
'''
with smart_open.smart_open('s3://predictd/Observed/training_data.encodePilots_and_nchars.rdd.data_idx.pickle', 'rb') as pickle_in:
    data_idx = pickle.loads(pickle_in.read())
with smart_open.smart_open('s3://predictd/reference_data/folds.5.8.pickle', 'rb') as pickle_in:
    folds = pickle.loads(pickle_in.read())

assay_mat_coords = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K4me1', 'H3K27ac', 'DNase'])
ct_mat_coords = set((elt[0][:20], elt[-1][0]) for elt in data_idx.values())
ct_list, assay_coords = list(zip(*itertools.chain(*[[(ct_lab, (ct_coord * 24) + assay_coord) for assay_coord in assay_mat_coords]
                                                    for ct_lab, ct_coord in ct_mat_coords])))
ct_set = sorted(set((elt[0], elt[-1][0]) for elt in data_idx.values()), key=lambda x:x[1])
obs_coords = numpy.logical_or(~folds[0]['train'][0]['train'], numpy.logical_or(~folds[0]['train'][0]['valid'], ~folds[0]['test'])).flatten()[list(assay_coords)]
har_to_plot_noimp = har_data[:,assay_coords]
har_to_plot_noimp[:,~obs_coords] = 0
har_to_plot_noimp = sps.csr_matrix(har_to_plot_noimp)
print(har_to_plot_noimp.shape)


# In[7]:

print('Observed and imputed data in the same order: {!s}'.format(numpy.all([e1 == e2 for e1, e2 in zip(har_info, har_info_imp)])))


# Set colors for the Enhancer Finder tissue assignments. Use the order of har_types to sort within kmeans groups in the ensuing analyses.

# In[8]:

'''Print color key for Capra, et al. 2013 ncHAR cluster identities.
'''
colors = ["#9b9b9b",
"#593086",
"#ffea7e",
"#99ac45",
"#00641e",
"#21bd42",
"#5cc482",
"#903026",
"#5485d5",
"#d775c2",
"#001e9b",
"#c5953d",
"#2eb7e8",
"#d67344",
"#8c2b6b",
"#ed3b50"]

har_types = sorted(set(elt[-1] for elt in har_info))
har_counts = [len([elt for elt in har_info if elt[-1] == ht]) for ht in har_types]
har_labels = ['{!s} ({!s})'.format(ht, hc) for ht, hc in zip(har_types, har_counts)]
sort_key = sorted(range(len(har_types)), reverse=True, key=lambda x: har_counts[x])
har_types = numpy.array(har_types)[sort_key]
har_counts = numpy.array(har_counts)[sort_key]
har_labels = numpy.array(har_labels)[sort_key]
colors = numpy.array(colors)[sort_key]

fig = pyplot.figure(figsize=(5,8))
ax = fig.add_axes([0, 0, 1, 1])
cmap = matplotlib.colors.ListedColormap(colors)
bounds = numpy.arange(17)
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors))
feat_im = ax.matshow(numpy.arange(len(har_types))[:,None], cmap=cmap, norm=norm, origin='lower', aspect=0.12)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks(numpy.arange(har_types.shape[0]))
ax.set_yticklabels([])

for idx, har_label in enumerate(har_labels):
#    ax.text(0.5*(start+end), height, displaystring, 
#             ha='center', va='center', 
#             bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),
#             size=fontsize)
    color = '#f2f2f2' if idx in [1,3,4,6,8,9,12] else 'k'
    ax.text(0, idx, har_label.replace(';', '; ').replace('_', ' '),
            ha='center', va='center', size=16, fontweight='bold', color=color)
fig.show()
fig.savefig('../../predictd_figs/capra2013_cluster_colors.pdf', bbox_inches='tight')
#print('\n'.join(har_types))


# Make vector of EnhancerFinder tissue type assignments for the har_data and har_data_imp rows

# In[9]:

'''Save Capra, et al. 2013 ncHAR cluster identities for matching up with our clusters later.
'''
har_type_assignments = numpy.zeros(len(har_info))
har_types_array = numpy.array([elt[-1] for elt in har_info])
for idx, htype in enumerate(har_types):
    htype_idx = numpy.where(har_types_array == htype)[0]
    har_type_assignments[htype_idx] = idx


# ##Biclustering Analysis
# ###Obs Only

# In[10]:

'''Extract the relevant assays from the total data.
'''
h3k4me1_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K4me1']).pop()
h3k27ac_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K27ac']).pop()
dnase_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['DNase']).pop()

imp_coords = ~numpy.logical_or(~folds[0]['train'][0]['train'], numpy.logical_or(~folds[0]['train'][0]['valid'], ~folds[0]['test'])).flatten()
har_data_only_obs = har_data.copy()
har_data_only_obs[:,imp_coords] = 0
data_by_assay = [[],[],[]]
for ct, ct_mat_coord in ct_set:
    coord = (ct_mat_coord * 24) + h3k4me1_mat_coord
    data_by_assay[0].extend(har_data_only_obs[:,coord].flatten())
    coord = (ct_mat_coord * 24) + h3k27ac_mat_coord
    data_by_assay[1].extend(har_data_only_obs[:,coord].flatten())
    coord = (ct_mat_coord * 24) + dnase_mat_coord
    data_by_assay[2].extend(har_data_only_obs[:,coord].flatten())
data_by_assay = sps.csr_matrix(numpy.array(data_by_assay))
print(data_by_assay.shape)


# In[11]:

'''Use PCA to reduce the dimensionality of the assay dimension (left with ncHARs and cell types)
'''
by_assay_pca = TruncatedSVD(n_components=1, random_state=11234).fit_transform(data_by_assay.T)
print(by_assay_pca.shape)


# In[12]:

'''Convert the first principal component back into typical (127,24) array shape
'''
har_data_assay_collapsed = by_assay_pca.reshape(127, -1).T
print(har_data_assay_collapsed.shape)


# In[13]:

'''Compute BIC and silhouette score for k=2 to k=40 so we can choose the number of ncHAR clusters
Note that this was purely exploratory; in the end we picked the number of cell type clusters based on 
this analysis done on the imputed data, not observed data.
'''
from sklearn.metrics import silhouette_score

model_bic = []
model_silhouette = []
num_components = numpy.arange(2,40)
for ncomp in num_components:  
    model = KMeans(n_clusters=ncomp, random_state=927).fit(har_data_assay_collapsed)
    labels = model.labels_
    model_bic.append(compute_bic(model, har_data_assay_collapsed))
    model_silhouette.append(silhouette_score(har_data_assay_collapsed, labels, metric='euclidean'))


# In[14]:

'''Plot BIC and silhouette results
'''
pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})
fig, axes = pyplot.subplots(nrows=2, ncols=1, figsize=(8,16))
axes[0].plot(num_components, model_bic, color='b')
axes[0].set_ylabel('KMeans BIC', fontsize=16)
axes[0].set_xlabel('Number of ncHAR clusters', fontsize=16)
axes[0].axvline(5, color='r')

axes[1].plot(num_components, model_silhouette, color='b')
axes[1].set_ylabel('KMeans silhouette', fontsize=16)
axes[1].set_xlabel('Number of ncHAR clusters', fontsize=16)
axes[1].axvline(5, color='r')
fig.show()


# In[15]:

'''Compute BIC and silhouette score for k=2 to k=40 so we can choose the number of cell type clusters
Note that this was purely exploratory; in the end we picked the number of cell type clusters based on 
this analysis done on the imputed data, not observed data.
'''
from sklearn.metrics import silhouette_score

model_bic = []
model_silhouette = []
num_components = numpy.arange(2,40)
for ncomp in num_components:  
    model = KMeans(n_clusters=ncomp, random_state=927).fit(har_data_assay_collapsed.T)
    labels = model.labels_
    model_bic.append(compute_bic(model, har_data_assay_collapsed.T))
    model_silhouette.append(silhouette_score(har_data_assay_collapsed.T, labels, metric='euclidean'))


# In[16]:

'''Plot BIC and silhouette results
'''
pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})
fig, axes = pyplot.subplots(nrows=2, ncols=1, figsize=(8,16))
axes[0].plot(num_components, model_bic, color='b')
axes[0].set_ylabel('KMeans BIC', fontsize=16)
axes[0].set_xlabel('Number of cell type clusters', fontsize=16)
axes[0].axvline(6, color='r')

axes[1].plot(num_components, model_silhouette, color='b')
axes[1].set_ylabel('KMeans silhouette', fontsize=16)
axes[1].set_xlabel('Number of cell type clusters', fontsize=16)
axes[1].axvline(6, color='r')
fig.show()


# In[17]:

'''Compute biclustering of observed data
'''
#biclustering based on 5 ncHAR clusters (rows) and 6 cell type clusters (columns), based on the imputed data BIC and silhouette analysis
from sklearn.cluster import bicluster
biclust_obs = bicluster.SpectralBiclustering(n_clusters=(5,6), method='log', n_components=6, 
                                             n_best=3, svd_method='arpack', random_state=132).fit(har_data_assay_collapsed)


# In[18]:

'''Flatten biclustering into marginal clustering of each axis (one for ncHARs and one for cell types)
'''
cluster_locs = []
for idx in range(0, 30, 6):
    cluster_locs.append(numpy.where(biclust_obs.rows_[idx])[0])
cluster_locs.sort(key=lambda x: len(x), reverse=True)
har_clusters_obs = []
for idx, clust in enumerate(cluster_locs):
    har_clusters_obs.append(numpy.array([clust, (numpy.ones(clust.shape) * idx).astype(int)]))
har_clusters_obs = numpy.hstack(har_clusters_obs).T
print(har_clusters_obs.shape)

cluster_locs = []
for idx in range(6):
    cluster_locs.append(numpy.where(biclust_obs.columns_[idx])[0])
cluster_locs.sort(key=lambda x: len(x), reverse=True)
ct_clusters_obs = []
for idx, clust in enumerate(cluster_locs):
    ct_clusters_obs.append(numpy.array([clust, (numpy.ones(clust.shape) * idx).astype(int)]))
ct_clusters_obs = numpy.hstack(ct_clusters_obs).T
print(ct_clusters_obs.shape)


# In[19]:

'''Print the cell types in each cluster for easy examination.
'''
ct_list = numpy.array([elt[0] for elt in sorted(set((elt[0], elt[-1][0]) for elt in data_idx.values()), key=lambda x:x[1])])
for idx in range(6):
    ct_idx = ct_clusters_obs[numpy.where(ct_clusters_obs[:,1] == idx)[0],0].flatten()
    print('{!s}:\n{!s}\n'.format(idx, '\n'.join(ct_list[ct_idx])))


# In[20]:

'''Plot the observed data biclustering results as a heatmap.
'''
colors6 = ["#697cd4",
"#98a441",
"#9350a1",
"#50b47b",
"#ba496b",
"#ba6437"]

colors12 = ["#840000",
"#4da128",
"#7058ce",
"#dbb126",
"#17176f",
"#00813d",
"#c247b3",
"#ff9e7f",
"#014e9c",
"#b6003e",
"#0084c3",
"#f4b4ff"]

#sum column vectors corresponding to same cell type
ct_list = sorted(set((elt[0], elt[-1][0]) for elt in data_idx.values()), key=lambda x:x[1])
ct_list_array = numpy.array([elt[0] for elt in ct_list])
assay_list = sorted(set((elt[1], elt[-1][1]) for elt in data_idx.values()
                        if elt[1] in ['DNase', 'H3K4me1', 'H3K27ac']), key=lambda x:x[1])
to_plot = []
for ct, ct_coord in ct_list:
    cols = [assay_coords.index(elt) for elt in [(ct_coord * 24) + elt[1] for elt in assay_list]]
    to_plot.append(numpy.sum(numpy.arcsinh(har_to_plot_noimp.toarray()[:,cols]), axis=1))
to_plot = numpy.vstack(to_plot).T
#to_plot /= numpy.max(to_plot, axis=0)

matplotlib.rcParams.update({'font.size': 8})
fig = pyplot.figure(figsize=(22,20))

#get columns sort order
new_col_labels = numpy.zeros(biclust_obs.column_labels_.shape)
col_sort = []
for new_clust_idx, orig_clust_idx in enumerate(sorted(range(6), reverse=True, 
                                                      key=lambda x: len(numpy.where(biclust_obs.column_labels_ == x)[0]))):
    clust_coords = numpy.where(biclust_obs.column_labels_ == orig_clust_idx)[0]
    grouped_by_ct_name = numpy.argsort(ct_list_array[clust_coords])
    col_sort.extend(clust_coords[grouped_by_ct_name])
    new_col_labels[clust_coords] = new_clust_idx
col_sort = numpy.array(col_sort)
col_labels = [ct_list[i][0][:20] for i in col_sort]

#get rows sort order
new_row_labels = numpy.zeros(biclust_obs.row_labels_.shape)
row_sort = []
for new_clust_idx, orig_clust_idx in enumerate(sorted(range(6), reverse=True, 
                                                      key=lambda x: len(numpy.where(biclust_obs.row_labels_ == x)[0]))):
    clust_coords = numpy.where(biclust_obs.row_labels_ == orig_clust_idx)[0]
    grouped_by_htype = numpy.argsort(har_type_assignments[clust_coords])
    row_sort.extend(clust_coords[grouped_by_htype])
    new_row_labels[clust_coords] = new_clust_idx
row_sort = numpy.array(row_sort)

#ct cluster colors (x-axis)
ax = fig.add_axes([0.1, 0.91, 0.64, 0.099])
cmap = matplotlib.colors.ListedColormap(colors6)
bounds = numpy.cumsum([0] + [numpy.sum(new_col_labels == idx) for idx in range(6)])
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors6))
feat_im = ax.matshow(numpy.arange(new_col_labels.shape[0])[None,:], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xticks(numpy.arange(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=90, fontsize=8)

#har cluster colors (y-axis)
ax2 = fig.add_axes([0.045, 0, 0.053, 0.9])
cmap = matplotlib.colors.ListedColormap(colors12)
bounds = numpy.cumsum([0] + [numpy.sum(new_row_labels == idx) for idx in range(6)])
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors12[:6]))
feat_im = ax2.matshow(numpy.arange(new_row_labels.shape[0])[:,None], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax2.set_xticks([])
ax2.set_xticklabels([])
ax2.set_yticks([])
ax2.set_yticklabels([])

#har predicted-tissue activity colors (y-axis2)
ax3 = fig.add_axes([0, 0, 0.043, 0.9])
cmap = matplotlib.colors.ListedColormap(colors)
bounds = numpy.arange(16)
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors))
feat_im = ax3.matshow(har_type_assignments[row_sort,None], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax3.set_xticks([])
ax3.set_xticklabels([])

#plot sum of h3k4me1, h3k27ac, dnase signal for each HAR and cell type
axes = fig.add_axes([0.1, 0, 0.8, 0.9])
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(to_plot), numpy.max(to_plot), 20), pyplot.cm.viridis.N)
to_plot_sort = to_plot[:,col_sort]
im = axes.matshow(to_plot_sort[row_sort,:], cmap='viridis', origin='lower', norm=norm, aspect='auto')
axes.set_xticks([])
axes.set_yticks([])

fig.colorbar(im)

#print(len(row_sort))
#rint(adjusted_rand_score(har_type_assignments[row_sort], har_clusters[:,1]))

fig.show()
fig.savefig('../../predictd_figs/obs_nchar_heatmap.pdf', bbox_inches='tight')


# ###Imp Only

# In[21]:

'''Extract the relevant assays from the total data.
'''
h3k4me1_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K4me1']).pop()
h3k27ac_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K27ac']).pop()
dnase_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['DNase']).pop()

data_by_assay = [[],[],[]]
for ct, ct_mat_coord in ct_set:
    coord = (ct_mat_coord * 24) + h3k4me1_mat_coord
    data_by_assay[0].extend(har_data_imp[:,coord].flatten())
    coord = (ct_mat_coord * 24) + h3k27ac_mat_coord
    data_by_assay[1].extend(har_data_imp[:,coord].flatten())
    coord = (ct_mat_coord * 24) + dnase_mat_coord
    data_by_assay[2].extend(har_data_imp[:,coord].flatten())
data_by_assay = sps.csr_matrix(numpy.array(data_by_assay))
print(data_by_assay.shape)
print(numpy.sum(data_by_assay.toarray() == 0))


# In[22]:

'''Use PCA to reduce the dimensionality of the assay dimension (left with ncHARs and cell types)
'''
by_assay_pca = TruncatedSVD(n_components=2, random_state=114).fit_transform(data_by_assay.T)
print(by_assay_pca.shape)


# In[23]:

'''Convert the first principal component back into typical (127,24) array shape
'''
har_data_assay_collapsed = by_assay_pca[:,0].reshape(127, -1).T
print(har_data_assay_collapsed.shape)
print(numpy.sum(har_data_assay_collapsed == 0))


# In[24]:

'''Compute BIC and silhouette score for k=2 to k=40 so we can choose the number of ncHAR clusters
'''
from sklearn.metrics import silhouette_score

model_bic = []
model_silhouette = []
num_components = numpy.arange(2,40)
for ncomp in num_components:  
    model = KMeans(n_clusters=ncomp, random_state=927).fit(har_data_assay_collapsed)
    labels = model.labels_
    model_bic.append(compute_bic(model, har_data_assay_collapsed))
    model_silhouette.append(silhouette_score(har_data_assay_collapsed, labels, metric='euclidean'))


# In[25]:

'''Plot BIC and silhouette results
'''
pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})
fig, axes = pyplot.subplots(nrows=2, ncols=1, figsize=(8,16))
axes[0].plot(num_components, model_bic, color='b')
axes[0].set_ylabel('KMeans BIC', fontsize=16)
axes[0].set_xlabel('Number of ncHAR clusters', fontsize=16)
axes[0].axvline(5, color='r')

axes[1].plot(num_components, model_silhouette, color='b')
axes[1].set_ylabel('KMeans silhouette', fontsize=16)
axes[1].set_xlabel('Number of ncHAR clusters', fontsize=16)
axes[1].axvline(5, color='r')
fig.show()
fig.savefig('../../predictd_figs/nchar_cluster_choice.pdf', bbox_inches='tight')


# In[26]:

'''Compute BIC and silhouette score for k=2 to k=40 so we can choose the number of cell type clusters
'''
from sklearn.metrics import silhouette_score

model_bic = []
model_silhouette = []
num_components = numpy.arange(2,40)
for ncomp in num_components:  
    model = KMeans(n_clusters=ncomp, random_state=927).fit(har_data_assay_collapsed.T)
    labels = model.labels_
    model_bic.append(compute_bic(model, har_data_assay_collapsed.T))
    model_silhouette.append(silhouette_score(har_data_assay_collapsed.T, labels, metric='euclidean'))


# In[27]:

'''Plot BIC and silhouette results
'''
fig, axes = pyplot.subplots(nrows=2, ncols=1, figsize=(8,16))
axes[0].plot(num_components, model_bic, color='b')
axes[0].set_ylabel('KMeans BIC', fontsize=16)
axes[0].set_xlabel('Number of cell type clusters', fontsize=16)
axes[0].axvline(6, color='r')

axes[1].plot(num_components, model_silhouette, color='b')
axes[1].set_ylabel('KMeans silhouette', fontsize=16)
axes[1].set_xlabel('Number of cell type clusters', fontsize=16)
axes[1].axvline(6, color='r')
fig.show()
fig.savefig('../../predictd_figs/cell_type_cluster_choice.pdf', bbox_inches='tight')


# In[28]:

'''Compute biclustering of imputed data
'''
#biclustering based on 5 ncHAR clusters (rows) and 6 cell type clusters (columns), based on the imputed data BIC and silhouette analysis
from sklearn.cluster import bicluster
biclust_imp = bicluster.SpectralBiclustering(n_clusters=(5,6), method='log', n_components=6, 
                                             n_best=3, svd_method='arpack', random_state=132).fit(har_data_assay_collapsed)


# In[29]:

'''Flatten biclustering into marginal clustering of each axis (one for ncHARs and one for cell types)
'''
cluster_locs = []
for idx in range(0,30,6):
    cluster_locs.append(numpy.where(biclust_imp.rows_[idx])[0])
cluster_locs.sort(key=lambda x: len(x), reverse=True)
har_clusters_imp = []
for idx, clust in enumerate(cluster_locs):
    har_clusters_imp.append(numpy.array([clust, (numpy.ones(clust.shape) * idx).astype(int)]))
har_clusters_imp = numpy.hstack(har_clusters_imp).T
print(har_clusters_imp.shape)

cluster_locs = []
for idx in range(6):
    cluster_locs.append(numpy.where(biclust_imp.columns_[idx])[0])
cluster_locs.sort(key=lambda x: len(x), reverse=True)
ct_clusters_imp = []
for idx, clust in enumerate(cluster_locs):
    ct_clusters_imp.append(numpy.array([clust, (numpy.ones(clust.shape) * idx).astype(int)]))
ct_clusters_imp = numpy.hstack(ct_clusters_imp).T
print(ct_clusters_imp.shape)


# In[30]:

'''Print the cell types in each cluster for easy examination.
'''
ct_list = numpy.array([elt[0] for elt in sorted(set((elt[0], elt[-1][0]) for elt in data_idx.values()), key=lambda x:x[1])])
for idx in range(6):
    ct_idx = ct_clusters_imp[numpy.where(ct_clusters_imp[:,1] == idx)[0],0].flatten()
    print('{!s}:\n{!s}\n'.format(idx, '\n'.join(ct_list[ct_idx])))


# In[31]:

'''Plot the imputed data biclustering results as a heatmap.
'''
colors6 = ["#697cd4",
"#98a441",
"#9350a1",
"#50b47b",
"#ba496b",
"#ba6437"]

colors12 = ["#840000",
"#4da128",
"#7058ce",
"#dbb126",
"#17176f",
"#00813d",
"#c247b3",
"#ff9e7f",
"#014e9c",
"#b6003e",
"#0084c3",
"#f4b4ff"]

#sum column vectors corresponding to same cell type
ct_list = sorted(set((elt[0], elt[-1][0]) for elt in data_idx.values()), key=lambda x:x[1])
ct_list_array = numpy.array([elt[0] for elt in ct_list])
assay_list = sorted(set((elt[1], elt[-1][1]) for elt in data_idx.values()
                        if elt[1] in ['DNase', 'H3K4me1', 'H3K27ac']), key=lambda x:x[1])
to_plot = []
for ct, ct_coord in ct_list:
    cols = [(ct_coord * 24) + elt[1] for elt in assay_list]
    to_plot.append(numpy.sum(numpy.arcsinh(har_data_imp[:,cols]), axis=1))
to_plot = numpy.vstack(to_plot).T

matplotlib.rcParams.update({'font.size': 8})
fig = pyplot.figure(figsize=(22,20))

har_clusters = har_clusters_imp
ct_clusters = ct_clusters_imp

#get columns sort order
new_col_labels = numpy.zeros(biclust_imp.column_labels_.shape)
col_sort = []
for new_clust_idx, orig_clust_idx in enumerate(sorted(range(6), reverse=True, key=lambda x: len(numpy.where(biclust_imp.column_labels_ == x)[0]))):
    clust_coords = numpy.where(biclust_imp.column_labels_ == orig_clust_idx)[0]
    grouped_by_ct_name = numpy.argsort(ct_list_array[clust_coords])
    col_sort.extend(clust_coords[grouped_by_ct_name])
    new_col_labels[clust_coords] = new_clust_idx
col_sort = numpy.array(col_sort)
col_labels = [ct_list[i][0][:20] for i in col_sort]

#get rows sort order
new_row_labels = numpy.zeros(biclust_imp.row_labels_.shape)
row_sort = []
for new_clust_idx, orig_clust_idx in enumerate(sorted(range(6), reverse=True, key=lambda x: len(numpy.where(biclust_imp.row_labels_ == x)[0]))):
    clust_coords = numpy.where(biclust_imp.row_labels_ == orig_clust_idx)[0]
    grouped_by_htype = numpy.argsort(har_type_assignments[clust_coords])
    row_sort.extend(clust_coords[grouped_by_htype])
    new_row_labels[clust_coords] = new_clust_idx
row_sort = numpy.array(row_sort)

#ct cluster colors (x-axis)
ax = fig.add_axes([0.1, 0.91, 0.64, 0.099])
cmap = matplotlib.colors.ListedColormap(colors6)
bounds = numpy.cumsum([0] + [numpy.sum(new_col_labels == idx) for idx in range(6)])
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors6))
feat_im = ax.matshow(numpy.arange(new_col_labels.shape[0])[None,:], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xticks(numpy.arange(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=90, fontsize=8)

#har cluster colors (y-axis)
ax2 = fig.add_axes([0.045, 0, 0.053, 0.9])
cmap = matplotlib.colors.ListedColormap(colors12)
bounds = numpy.cumsum([0] + [numpy.sum(new_row_labels == idx) for idx in range(6)])
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors12[:6]))
feat_im = ax2.matshow(numpy.arange(new_row_labels.shape[0])[:,None], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax2.set_xticks([])
ax2.set_xticklabels([])
ax2.set_yticks([])
ax2.set_yticklabels([])

#har predicted-tissue activity colors (y-axis2)
ax3 = fig.add_axes([0, 0, 0.043, 0.9])
cmap = matplotlib.colors.ListedColormap(colors)
bounds = numpy.arange(16)
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors))
feat_im = ax3.matshow(har_type_assignments[row_sort,None], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax3.set_xticks([])
ax3.set_xticklabels([])

#plot sum of h3k4me1, h3k27ac, dnase signal for each HAR and cell type
axes = fig.add_axes([0.1, 0, 0.8, 0.9])
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(to_plot), numpy.max(to_plot), 20), pyplot.cm.viridis.N)
to_plot_sort = to_plot[:,col_sort]
im = axes.matshow(to_plot_sort[row_sort,:], cmap='viridis', origin='lower', norm=norm, aspect='auto')
axes.set_xticks([])
axes.set_yticks([])

fig.colorbar(im)

fig.show()
fig.savefig('../../predictd_figs/imp_nchar_heatmap.pdf', bbox_inches='tight')


# ###Shuffled ncHARs

# In[32]:

'''Extract the relevant assays from the total data.
'''
h3k4me1_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K4me1']).pop()
h3k27ac_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K27ac']).pop()
dnase_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['DNase']).pop()

imp_coords = ~numpy.logical_or(~folds[0]['train'][0]['train'], numpy.logical_or(~folds[0]['train'][0]['valid'], ~folds[0]['test'])).flatten()
har_data_rand_only_obs = har_data_rand.copy()
har_data_rand_only_obs[:,imp_coords] = 0
data_by_assay = [[],[],[]]
for ct, ct_mat_coord in ct_set:
    coord = (ct_mat_coord * 24) + h3k4me1_mat_coord
    data_by_assay[0].extend(har_data_rand_only_obs[:,coord].flatten())
    coord = (ct_mat_coord * 24) + h3k27ac_mat_coord
    data_by_assay[1].extend(har_data_rand_only_obs[:,coord].flatten())
    coord = (ct_mat_coord * 24) + dnase_mat_coord
    data_by_assay[2].extend(har_data_rand_only_obs[:,coord].flatten())
data_by_assay = sps.csr_matrix(numpy.array(data_by_assay))
print(data_by_assay.shape)
print(numpy.sum(data_by_assay.toarray() == 0))


# In[33]:

'''Use PCA to reduce the dimensionality of the assay dimension (left with ncHARs and cell types)
'''
by_assay_pca = TruncatedSVD(n_components=2, random_state=114).fit_transform(data_by_assay.T)
print(by_assay_pca.shape)


# In[34]:

'''Convert the first principal component back into typical (127,24) array shape
'''
har_data_assay_collapsed = by_assay_pca[:,0].reshape(127, -1).T
print(har_data_assay_collapsed.shape)
print(numpy.sum(har_data_assay_collapsed == 0))


# In[35]:

'''Compute biclustering of "shuffled ncHAR coordinates" negative control
'''
#biclustering based on 5 ncHAR clusters (rows) and 6 cell type clusters (columns), based on the imputed data BIC and silhouette analysis
from sklearn.cluster import bicluster
biclust_rand = bicluster.SpectralBiclustering(n_clusters=(5,6), method='log', n_components=6, 
                                             n_best=3, svd_method='arpack', random_state=132).fit(har_data_assay_collapsed)


# In[36]:

'''Flatten biclustering into marginal clustering of each axis (one for ncHARs and one for cell types)
'''
cluster_locs = []
for idx in range(0, 30, 6):
    cluster_locs.append(numpy.where(biclust_rand.rows_[idx])[0])
cluster_locs.sort(key=lambda x: len(x), reverse=True)
rand_har_clusters = []
for idx, clust in enumerate(cluster_locs):
    rand_har_clusters.append(numpy.array([clust, (numpy.ones(clust.shape) * idx).astype(int)]))
rand_har_clusters = numpy.hstack(rand_har_clusters).T
print(rand_har_clusters.shape)

cluster_locs = []
for idx in range(6):
    cluster_locs.append(numpy.where(biclust_rand.columns_[idx])[0])
cluster_locs.sort(key=lambda x: len(x), reverse=True)
rand_ct_clusters = []
for idx, clust in enumerate(cluster_locs):
    rand_ct_clusters.append(numpy.array([clust, (numpy.ones(clust.shape) * idx).astype(int)]))
rand_ct_clusters = numpy.hstack(rand_ct_clusters).T
print(rand_ct_clusters.shape)


# ###Other assays

# In[37]:

'''Extract the relevant assays from the total data.
'''
h3k4me3_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K4me3']).pop()
h3k27me3_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K27me3']).pop()
h3k36me3_mat_coord = set(elt[-1][1] for elt in data_idx.values() if elt[1] in ['H3K36me3']).pop()

data_by_assay = [[],[],[]]
for ct, ct_mat_coord in ct_set:
    coord = (ct_mat_coord * 24) + h3k4me3_mat_coord
    data_by_assay[0].extend(har_data[:,coord].flatten())
    coord = (ct_mat_coord * 24) + h3k27me3_mat_coord
    data_by_assay[1].extend(har_data[:,coord].flatten())
    coord = (ct_mat_coord * 24) + h3k36me3_mat_coord
    data_by_assay[2].extend(har_data[:,coord].flatten())
data_by_assay = sps.csr_matrix(numpy.array(data_by_assay))
print(data_by_assay.shape)
print(numpy.sum(data_by_assay.toarray() == 0))


# In[38]:

'''Use PCA to reduce the dimensionality of the assay dimension (left with ncHARs and cell types)
'''
by_assay_pca = TruncatedSVD(n_components=2, random_state=114).fit_transform(data_by_assay.T)
print(by_assay_pca.shape)


# In[39]:

'''Convert the first principal component back into typical (127,24) array shape
'''
har_data_assay_collapsed = by_assay_pca[:,0].reshape(127, -1).T
print(har_data_assay_collapsed.shape)
print(numpy.sum(har_data_assay_collapsed == 0))


# In[40]:

'''Compute biclustering of "other assays" negative control
'''
#biclustering based on 5 ncHAR clusters (rows) and 6 cell type clusters (columns), based on the imputed data BIC and silhouette analysis
from sklearn.cluster import bicluster
biclust_other = bicluster.SpectralBiclustering(n_clusters=(5,6), method='log', n_components=6, 
                                             n_best=3, svd_method='arpack', random_state=132).fit(har_data_assay_collapsed)


# In[41]:

'''Flatten biclustering into marginal clustering of each axis (one for ncHARs and one for cell types)
'''
cluster_locs = []
for idx in range(0, 30, 6):
    cluster_locs.append(numpy.where(biclust_other.rows_[idx])[0])
cluster_locs.sort(key=lambda x: len(x), reverse=True)
other_har_clusters = []
for idx, clust in enumerate(cluster_locs):
    other_har_clusters.append(numpy.array([clust, (numpy.ones(clust.shape) * idx).astype(int)]))
other_har_clusters = numpy.hstack(other_har_clusters).T
print(rand_har_clusters.shape)

cluster_locs = []
for idx in range(6):
    cluster_locs.append(numpy.where(biclust_other.columns_[idx])[0])
cluster_locs.sort(key=lambda x: len(x), reverse=True)
other_ct_clusters = []
for idx, clust in enumerate(cluster_locs):
    other_ct_clusters.append(numpy.array([clust, (numpy.ones(clust.shape) * idx).astype(int)]))
other_ct_clusters = numpy.hstack(other_ct_clusters).T
print(other_ct_clusters.shape)


# ###Evaluate Biclusters

# In[42]:

'''Compute the adjusted rand score between the different clusterings in order to assess how similar the solutions are.
'''
har_comp = [adjusted_rand_score(biclust_obs.row_labels_, biclust_imp.row_labels_),
            adjusted_rand_score(biclust_imp.row_labels_, har_type_assignments),
            adjusted_rand_score(biclust_obs.row_labels_, har_type_assignments),
            adjusted_rand_score(biclust_imp.row_labels_, biclust_rand.row_labels_),
            adjusted_rand_score(biclust_obs.row_labels_, biclust_rand.row_labels_),
            adjusted_rand_score(har_type_assignments, biclust_rand.row_labels_),
            adjusted_rand_score(biclust_imp.row_labels_, biclust_other.row_labels_),
            adjusted_rand_score(biclust_obs.row_labels_, biclust_other.row_labels_),
            adjusted_rand_score(har_type_assignments, biclust_other.row_labels_)]

ct_comp = [adjusted_rand_score(biclust_obs.column_labels_, biclust_imp.column_labels_),
           adjusted_rand_score(biclust_imp.column_labels_, biclust_rand.column_labels_),
           adjusted_rand_score(biclust_obs.column_labels_, biclust_rand.column_labels_),
           adjusted_rand_score(biclust_imp.column_labels_, biclust_other.column_labels_),
           adjusted_rand_score(biclust_obs.column_labels_, biclust_other.column_labels_)]
print(har_comp, ct_comp)


# In[43]:

'''Plot the adjusted Rand score results.
'''
pyplot.rcParams.update({#'xtick.labelsize':16,
                        'ytick.labelsize':16})

fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16,8), sharey=True, gridspec_kw={'width_ratios':[1.5,1]})

width = 0.75
xlocs_har = numpy.arange(len(har_comp))
xlabels_har = ['Imputed vs\nObserved', 'Imputed vs\nCapra, et al.', 'Observed vs\nCapra, et al.', 
               'Imputed vs\nShuffled', 'Observed vs\nShuffled', 'Capra, et al. vs\nShuffled', 'Imputed vs\nOther', 'Observed vs\nOther',
               'Capra, et al. vs\nOther']
xlocs_ct = numpy.arange(len(ct_comp))
xlabels_ct = ['Imputed vs\nObserved', 'Imputed vs\nShuffled', 'Observed vs\nShuffled', 'Imputed vs\nOther', 'Observed vs\nOther']

axes[0].bar(xlocs_har, har_comp, width=width, color='blue', align='center')
axes[0].set_title('ncHAR clustering', fontsize=24)
axes[0].set_ylabel('Adjusted Rand index', fontsize=22)
axes[0].set_xticks(xlocs_har)
axes[0].set_xticklabels(xlabels_har, fontsize=18, rotation=55)
axes[0].set_ylim(0,1)
axes[0].set_xmargin(0.025)
axes[0].autoscale_view(tight=True, scalex=True)

width=0.75
axes[1].bar(xlocs_ct, ct_comp, width=width, color='red', align='center')
axes[1].set_title('Cell type clustering', fontsize=24)
axes[1].set_ylabel('Adjusted Rand index', fontsize=22)
axes[1].set_xticks(xlocs_ct)
axes[1].set_xticklabels(xlabels_ct, fontsize=18, rotation=55)
axes[1].set_ylim(0,1)
axes[1].set_xmargin(0.05)
axes[1].autoscale_view(tight=True, scalex=True)

fig.tight_layout()
fig.savefig('../../predictd_figs/nchar_cluster_eval.pdf', bbox_inches='tight')


# ##GO Analysis

# In[44]:

'''Save the ncHAR cluster information to a file.
'''
#ncHARS
#output: har_info, imputed_cluster_idx, imputed_cluster_name, capra_cluster_idx, capra_cluster_name, observed_cluster_idx
cluster_info = numpy.zeros((len(har_info), 5)).astype(object)

har_label_map = {0: 'No signal',
                 1: 'Brain/ES',
                 2: 'Epithelial/Mesenchymal',
                 3: 'Non-immune',
                 4: 'Immune'}

new_row_labels = numpy.zeros(biclust_imp.row_labels_.shape)
for new_clust_idx, orig_clust_idx in enumerate(sorted(range(5), reverse=True, key=lambda x: len(numpy.where(biclust_imp.row_labels_ == x)[0]))):
    clust_coords = numpy.where(biclust_imp.row_labels_ == orig_clust_idx)[0]
    new_row_labels[clust_coords] = new_clust_idx

for idx in range(5):
    clust_rows = numpy.where(new_row_labels == idx)[0]
    cluster_info[clust_rows,0] = idx
    cluster_info[clust_rows,1] = har_label_map[idx]

for idx, htype in enumerate(har_types):
    clust_rows = numpy.where(har_type_assignments == idx)[0]
    cluster_info[clust_rows,2] = idx
    cluster_info[clust_rows,3] = htype

new_obs_row_labels = numpy.zeros(biclust_obs.row_labels_.shape)
for new_clust_idx, orig_clust_idx in enumerate(sorted(range(5), reverse=True, key=lambda x: len(numpy.where(biclust_obs.row_labels_ == x)[0]))):
    clust_coords = numpy.where(biclust_obs.row_labels_ == orig_clust_idx)[0]
    new_obs_row_labels[clust_coords] = new_clust_idx

for idx in range(5):
    clust_rows = numpy.where(new_obs_row_labels == idx)[0]
    cluster_info[clust_rows,4] = idx

#SET THIS AS DESIRED
out_dir = os.getcwd()
try:
    os.makedirs(out_dir)
except:
    pass
fname = os.path.join(out_dir, 'ncHAR_cluster_info.tab')
headers = ['chrom', 'start', 'stop', 'ncHAR name', 'predicted developmental enhancer activity (Capra et al. 2013)', 
           'predicted tissue (Capra et al. 2013)', 'imputed data cluster idx', 'imputed data cluster name', 'capra cluster idx', 
           'capra cluster name', 'observed data cluster idx']
numpy.savetxt(fname, numpy.hstack([numpy.array(har_info).astype(object), cluster_info]), header=', '.join(headers), fmt='%s', delimiter='\t')


# In[45]:

'''Save the ncHARs in each cluster to bed files for further analysis.
'''
har_info_array = numpy.array(har_info)

for idx in range(5):
    clust_info = har_info_array[numpy.where(new_row_labels == idx)[0]]
    print('{!s}: {!s}'.format(idx, clust_info.shape[0]))
    with open(os.path.join(out_dir, 'har_cluster.{!s}.bed'.format(idx)), 'w') as out:
        out.write('\n'.join(['\t'.join(clust_info[i]) for i in xrange(clust_info.shape[0])]) + '\n')


# In[46]:

counts = {'na':0, 'other':0}
for clust_idx in range(1,5):
    clust_rows = numpy.where(new_row_labels == clust_idx)[0]
    counts['na'] += numpy.sum(har_type_assignments[clust_rows] == 0)
    counts['other'] += numpy.sum(har_type_assignments[clust_rows] == 1)
print('ncHARs from non-specific EnhancerFinder categories that were not in our no signal category: {!s}'.format(counts))


# In[47]:

counts = 0
clust_rows = numpy.where(new_row_labels == 0)[0]
for i in range(1,16):
    counts += numpy.sum(har_type_assignments[clust_rows] == i)
print('ncHARs called as enhancers by EnhancerFinder are in our no signal category: {!s}'.format(counts))


# In[48]:

'''Print key for heatmap figure.
'''
clust_num = 5

fig = pyplot.figure(figsize=(3,5))
ax = fig.add_axes([0, 0, 1, 1])
cmap = matplotlib.colors.ListedColormap(colors12)
bounds = numpy.arange(len(colors12[:clust_num]) + 1)
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors12[:clust_num]))
feat_im = ax.matshow(numpy.arange(clust_num)[:,None], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks(numpy.arange(clust_num))
ax.set_yticklabels([])

label_map = {0: 'No signal',
             1: 'Brain/ES',
             2: 'Epithelial/\nMesenchymal',
             3: 'Non-immune',
             4: 'Immune'}
for idx in numpy.arange(clust_num):
    label = '{!s} ({!s})'.format(label_map[idx], len(numpy.where(new_row_labels == idx)[0]))
    color = '#f2f2f2' if idx in [0,1,2,4,5] else 'k'
    ax.text(0, idx, label, color=color,
            ha='center', va='center', size=16, fontweight='bold')

#ax.set_yticklabels(['{!s} ({!s})'.format(label_map[idx], len(numpy.where(har_kmeans_res == sorted_har_kmeans_clusters[idx])[0])) for idx in numpy.arange(clust_num)], fontsize=16)
fig.show()
fig.savefig('../../predictd_figs/nchar_cluster_colors.pdf', bbox_inches='tight')
#print('\n'.join(har_types))


# In[49]:

'''Save cell type cluster information to a file.
'''
#cell types
#output: cell type, imp_cluster_idx, imp_cluster_name, obs_cluster_idx
cluster_info = numpy.zeros((len(ct_list), 3)).astype(object)

ct_label_map = {0: 'Immune',
                1: 'Differentiated',
                2: 'Brain/Embryonic Stem Cell',
                3: 'Mesenchymal',
                4: 'Epithelial',
                5: 'Cultured Cells'}

new_col_labels = numpy.zeros(biclust_imp.column_labels_.shape)
for new_clust_idx, orig_clust_idx in enumerate(sorted(range(6), reverse=True, key=lambda x: len(numpy.where(biclust_imp.column_labels_ == x)[0]))):
    clust_coords = numpy.where(biclust_imp.column_labels_ == orig_clust_idx)[0]
    new_col_labels[clust_coords] = new_clust_idx

for clust_idx in range(6):
    clust_rows = numpy.where(new_col_labels == clust_idx)[0]
    cluster_info[clust_rows,0] = clust_idx
    cluster_info[clust_rows,1] = ct_label_map[clust_idx]

new_col_labels = numpy.zeros(biclust_obs.column_labels_.shape)
for new_clust_idx, orig_clust_idx in enumerate(sorted(range(6), reverse=True, key=lambda x: len(numpy.where(biclust_obs.column_labels_ == x)[0]))):
    clust_coords = numpy.where(biclust_obs.column_labels_ == orig_clust_idx)[0]
    new_col_labels[clust_coords] = new_clust_idx

for clust_idx in range(6):
    clust_rows = numpy.where(new_col_labels == clust_idx)[0]
    cluster_info[clust_rows,2] = clust_idx

#SET THIS AS DESIRED
out_dir = os.getcwd()
try:
    os.makedirs(out_dir)
except:
    pass
fname = os.path.join(out_dir, 'ct_cluster_info.tab')
headers = ['cell type', 'imputed data cluster idx', 'imputed data cluster name', 'observed data cluster idx']
numpy.savetxt(fname, numpy.hstack([numpy.array(ct_list).astype(object)[:,0][:,None], cluster_info]), header=', '.join(headers), fmt='%s', delimiter='\t')


# In[49]:



