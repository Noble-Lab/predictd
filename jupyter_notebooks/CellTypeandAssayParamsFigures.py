
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import copy
import glob
import gzip
import matplotlib
from matplotlib import pyplot
import numpy
import os
import pickle
import random
import scipy
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
import smart_open
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


### Load Cell Type Parameters and Labels

# In[3]:

ct_params_url = 's3://predictd/PREDICTD/trained_predictd_models/test0/valid_fold0/ct_factors.pickle'
with smart_open.smart_open(ct_params_url, 'rb') as pickle_in:
    ct_params_tmp = pickle.loads(pickle_in.read())
#ct_params = numpy.hstack([ct_params_tmp[1][:,None], ct_params_tmp[0]])
ct_params = ct_params_tmp[0]

with smart_open.smart_open('s3://predictd/Observed/training_data.encodePilots_and_nchars.rdd.data_idx.pickle', 'rb') as pickle_in:
    data_idx = pickle.loads(pickle_in.read())
ct_labels = [ct[0].replace('_', ' ') for ct in 
             sorted(set([(elt[0],elt[-1][0]) for elt in data_idx.values()]),
                    key=lambda x:x[1])]
print('Num Cell Types: {!s}'.format(len(ct_labels)))
print('Shape of Cell Type Params: {!s}'.format(ct_params.shape))


### Hierarchical Cell Type Clustering

#### Cosine Distance, Average Clustering

# In[4]:

pyplot.rcParams.update({'xtick.labelsize':18,
                        'ytick.labelsize':18})

fig = pyplot.figure(figsize=(30,30))

#dendrogram plot for cts
ax1 = fig.add_axes([0.735, 0.02, 0.055, 0.95])
dist = distance.squareform(distance.pdist(ct_params, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
Z1 = hierarchy.dendrogram(Y, orientation='right', no_labels=True, ax=ax1, count_sort=True)
ax1.set_xticks([])
ax1.set_xticklabels([])

#dendrogram plot for lfs
ax2 = fig.add_axes([0, 0.975, 0.73, 0.025])
dist = distance.squareform(distance.pdist(ct_params.T, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, ax=ax2)
ax2.set_yticks([])
ax2.set_yticklabels([])

#LF matrix plot
ax3 = fig.add_axes([0, 0.02, 0.73, 0.95])
ct_index = Z1['leaves']
lf_index = Z2['leaves']
resort_ct_params = ct_params[ct_index,:]
resort_ct_params = resort_ct_params[:,lf_index]
im = ax3.matshow(resort_ct_params, cmap='viridis', origin='lower', aspect='auto')
ax3.set_xticks([])

#left, bottom, width, height
#cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
cbaxes = fig.add_axes([0, 0, 0.73, 0.015]) 
cb = fig.colorbar(im, orientation='horizontal', cax=cbaxes)

ax3.set_yticks(numpy.arange(ct_params.shape[0]))
ax3.set_yticklabels(numpy.array(ct_labels)[ct_index])
fig.show()


# ###No latent factor dendrogram (more room for cell type labels)

# In[5]:

pyplot.rcParams.update({'xtick.labelsize':20,
                        'ytick.labelsize':20})

fig = pyplot.figure(figsize=(30,30))

#dendrogram plot for cts
ax1 = fig.add_axes([0.735, 0.02, 0.055, 0.985])
dist = distance.squareform(distance.pdist(ct_params, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
Z1 = hierarchy.dendrogram(Y, orientation='right', no_labels=True, ax=ax1, count_sort=True)
ax1.set_xticks([])
ax1.set_xticklabels([])

#dendrogram plot for lfs
#ax2 = fig.add_axes([0, 0.975, 0.73, 0.025])
dist = distance.squareform(distance.pdist(ct_params.T, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
#Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, ax=ax2)
Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, no_plot=True)
#ax2.set_yticks([])
#ax2.set_yticklabels([])

#LF matrix plot
ax3 = fig.add_axes([0, 0.02, 0.73, 0.985])
ct_index = Z1['leaves']
lf_index = Z2['leaves']
resort_ct_params = ct_params[ct_index,:]
resort_ct_params = resort_ct_params[:,lf_index]
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(resort_ct_params), numpy.max(resort_ct_params), 20), pyplot.cm.viridis.N)
im = ax3.matshow(resort_ct_params, cmap='viridis', origin='lower', norm=norm, aspect='auto')
ax3.set_xlabel('Latent factors', fontsize=28)
ax3.xaxis.set_label_position('top')
ax3.xaxis.tick_top()
#ax3.set_xticks([])

#left, bottom, width, height
#cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
#cbaxes = fig.add_axes([0, 0, 0.73, 0.015]) 
cbaxes = fig.add_axes([0.12, 0, 0.5, 0.015]) 
cb = fig.colorbar(im, orientation='horizontal', cax=cbaxes)
cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=45)
cb.ax.set_xlabel('Latent factor value', fontsize=26)

ax3.set_yticks(numpy.arange(ct_params.shape[0]))
ax3.set_yticklabels(numpy.array(ct_labels)[ct_index])
fig.show()
fig.savefig('../../predictd_figs/ct_latent_factors_all.pdf', bbox_inches='tight')


# ##Zoomed

# In[6]:

numpy.where(numpy.array(ct_labels)[ct_index] == 'Brain Mid Frontal Lobe')


# In[7]:

pyplot.rcParams.update({'xtick.labelsize':26,
                        'ytick.labelsize':26})

fig = pyplot.figure(figsize=(30,20))

#dendrogram plot for cts
ax1 = fig.add_axes([0.735, 0.02, 0.055, 0.985])
dist = distance.squareform(distance.pdist(ct_params, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
Z1 = hierarchy.dendrogram(Y, orientation='right', no_labels=True, count_sort=True, no_plot=True)

dist2 = distance.squareform(distance.pdist(ct_params[Z1['leaves'][:62]], metric='cosine'))
Y2 = hierarchy.linkage(dist2, method='average')
Z12 = hierarchy.dendrogram(Y2, orientation='right', no_labels=True, count_sort=True, ax=ax1)

ax1.set_xticks([])
ax1.set_xticklabels([])

#dendrogram plot for lfs
#ax2 = fig.add_axes([0, 0.975, 0.73, 0.025])
dist = distance.squareform(distance.pdist(ct_params.T, metric='euclidean'))
Y = hierarchy.linkage(dist, method='ward')
#Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, ax=ax2)
Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, no_plot=True)
#ax2.set_yticks([])
#ax2.set_yticklabels([])

#LF matrix plot
ax3 = fig.add_axes([0, 0.02, 0.73, 0.985])
ct_index1 = Z1['leaves'][:62]
ct_index2 = Z12['leaves']
lf_index = Z2['leaves']
resort_ct_params = ct_params[ct_index1,:][ct_index2,:]
resort_ct_params = resort_ct_params[:,lf_index]
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(resort_ct_params), numpy.max(resort_ct_params), 20), pyplot.cm.viridis.N)
im = ax3.matshow(resort_ct_params, cmap='viridis', origin='lower', norm=norm, aspect='auto')
ax3.set_xlabel('Latent factors', fontsize=28)
ax3.xaxis.set_label_position('top')
ax3.xaxis.tick_top()

#left, bottom, width, height
#cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
#cbaxes = fig.add_axes([0, 0, 0.73, 0.015]) 
cbaxes = fig.add_axes([0.12, 0, 0.5, 0.015]) 
cb = fig.colorbar(im, orientation='horizontal', cax=cbaxes)
cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=45)
cb.ax.set_xlabel('Latent factor value', fontsize=26)

ax3.set_yticks(numpy.arange(resort_ct_params.shape[0]))
ax3.set_yticklabels(numpy.array(ct_labels)[ct_index1][ct_index2])
fig.show()
fig.savefig('../../predictd_figs/ct_latent_factors_zoom.pdf', bbox_inches='tight')


# ##Evaluate cell type clustering

# In[8]:

import sklearn
print(sklearn.__version__)


# In[9]:

dist = distance.squareform(distance.pdist(ct_params, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
#clust_assign = hierarchy.fcluster(Y, 1.15, criterion='inconsistent', depth=2, R=None, monocrit=None)
clust_assign = hierarchy.cut_tree(Y, n_clusters=8).flatten()
clust_val = metrics.calinski_harabaz_score(ct_params, clust_assign)
rand_vals = []
rs = numpy.random.RandomState(seed=5)
for i in xrange(1000):
    rand_vals.append(metrics.calinski_harabaz_score(ct_params, clust_assign[rs.permutation(ct_params.shape[0])]))


# In[10]:

pyplot.rcParams.update({'xtick.labelsize':20,
                        'ytick.labelsize':20})
fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8,8))
axes.hist(rand_vals, bins=15)
axes.axvline(clust_val, color='r', linestyle=':', linewidth=3)
axes.text(max(rand_vals) + 1.5, 160, 'Shuffled clusters\n$\mu=${:.2f} ($n=$1000)'.format(numpy.mean(rand_vals)), 
          color='b', size=20, ha='center', va='top')
axes.text(clust_val-1.3, 160, 'Real clusters\n$CHI=${:.2f}'.format(clust_val), 
          color='r', size=20, ha='center', va='top')
axes.set_xlabel('Calinski-Harabaz index', fontsize=22)
axes.set_ylabel('Count', fontsize=22)
axes.set_title('Cell type latent factor clustering\n', fontsize=22)
fig.show()
fig.savefig('../../predictd_figs/ct_lf_cluster_eval.pdf', bbox_inches='tight')


# ##Load Assay Data

# In[11]:

assay_params_url = 's3://predictd/PREDICTD/trained_predictd_models/test0/valid_fold0/assay_factors.pickle'
with smart_open.smart_open(assay_params_url, 'rb') as pickle_in:
    assay_params_tmp = pickle.loads(pickle_in.read())
#assay_params = numpy.hstack([assay_params_tmp[1][:,None], assay_params_tmp[0]])
assay_params = assay_params_tmp[0]

assay_labels = [assay[0] for assay in 
                 sorted(set([(elt[1],elt[-1][1]) for elt in data_idx.values()]),
                    key=lambda x:x[1])]
print('Num Assay Types: {!s}'.format(len(assay_labels)))
print('Shape of Assay Params: {!s}'.format(assay_params.shape))


# ##Cosine Distance, Average Clustering

# In[12]:

pyplot.rcParams.update({'xtick.labelsize':26,
                        'ytick.labelsize':26})

fig = pyplot.figure(figsize=(25,12))

#dendrogram plot for assays
ax1 = fig.add_axes([0.63, 0.03, 0.055, 0.82])
dist = distance.squareform(distance.pdist(assay_params, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
Z1 = hierarchy.dendrogram(Y, orientation='right', no_labels=True, ax=ax1, count_sort=True)
ax1.set_xticks([])
ax1.set_xticklabels([])

#dendrogram plot for lfs
ax2 = fig.add_axes([0, 0.86, 0.625, 0.055])
dist = distance.squareform(distance.pdist(assay_params.T, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, ax=ax2)
ax2.set_yticks([])
ax2.set_yticklabels([])

#LF matrix plot
ax3 = fig.add_axes([0, 0.03, 0.625, 0.82])
assay_index = Z1['leaves']
lf_index = Z2['leaves']
resort_assay_params = assay_params[assay_index,:]
resort_assay_params = resort_assay_params[:,lf_index]
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(resort_assay_params), 
                                                     numpy.max(resort_assay_params), 20), pyplot.cm.viridis.N)
im = ax3.matshow(resort_assay_params, cmap='viridis', origin='lower', norm=norm, aspect='auto')
ax3.set_xticks([])

#cbaxes = fig.add_axes([0, 0, 0.625, 0.02]) 
cbaxes = fig.add_axes([0.0875, 0, 0.45, 0.02]) 
cb = fig.colorbar(im, orientation='horizontal', cax=cbaxes)
cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=45)

ax3.set_yticks(numpy.arange(assay_params.shape[0]))
ax3.set_yticklabels(numpy.array(assay_labels)[assay_index])
fig.show()


# ###No latent factor dendrogram

# In[13]:

pyplot.rcParams.update({'xtick.labelsize':26,
                        'ytick.labelsize':26})

fig = pyplot.figure(figsize=(25,10))

#dendrogram plot for assays
ax1 = fig.add_axes([0.63, 0.035, 0.055, 0.85])
dist = distance.squareform(distance.pdist(assay_params, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
Z1 = hierarchy.dendrogram(Y, orientation='right', no_labels=True, ax=ax1, count_sort=True)
ax1.set_xticks([])
ax1.set_xticklabels([])

#dendrogram plot for lfs
#ax2 = fig.add_axes([0, 0.88, 0.62, 0.15])
dist = distance.squareform(distance.pdist(assay_params.T, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, no_plot=True)
#ax2.set_yticks([])
#ax2.set_yticklabels([])

#LF matrix plot
ax3 = fig.add_axes([0, 0.035, 0.625, 0.85])
assay_index = Z1['leaves']
lf_index = Z2['leaves']
resort_assay_params = assay_params[assay_index,:]
resort_assay_params = resort_assay_params[:,lf_index]
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(resort_assay_params), 
                                                     numpy.max(resort_assay_params), 20), pyplot.cm.viridis.N)
im = ax3.matshow(resort_assay_params, cmap='viridis', origin='lower', norm=norm, aspect='auto')
ax3.set_xlabel('Latent factors', fontsize=30)
ax3.xaxis.set_label_position('top')
ax3.xaxis.tick_top()

#cbaxes = fig.add_axes([0, 0, 0.625, 0.02]) 
cbaxes = fig.add_axes([0.0875, 0, 0.45, 0.02]) 
cb = fig.colorbar(im, orientation='horizontal', cax=cbaxes)
cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=45)
cb.ax.set_xlabel('Latent factor value', fontsize=28)

ax3.set_yticks(numpy.arange(assay_params.shape[0]))
ax3.set_yticklabels(numpy.array(assay_labels)[assay_index])
fig.show()
fig.savefig('../../predictd_figs/assay_latent_factors.pdf', bbox_inches='tight')


# ##Evaluate Assay Clustering

# In[14]:

dist = distance.squareform(distance.pdist(assay_params, metric='cosine'))
Y = hierarchy.linkage(dist, method='average')
clust_assign = hierarchy.cut_tree(Y, n_clusters=4).flatten()
clust_val = metrics.calinski_harabaz_score(assay_params, clust_assign)
rand_vals = []
rs = numpy.random.RandomState(seed=6)
for i in xrange(1000):
    rand_vals.append(metrics.calinski_harabaz_score(assay_params, clust_assign[rs.permutation(assay_params.shape[0])]))


# In[15]:

pyplot.rcParams.update({'xtick.labelsize':20,
                        'ytick.labelsize':20})
fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8,8))
axes.hist(rand_vals, bins=15)
axes.axvline(clust_val, color='r', linestyle=':', linewidth=3)
axes.text(max(rand_vals), 172, 'Shuffled clusters\n$\mu=${:.2f} ($n=$1000)'.format(numpy.mean(rand_vals)), 
          color='b', size=20, ha='center', va='top')
axes.text(clust_val-0.6, 172, 'Real clusters\n$CHI=${:.2f}'.format(clust_val), 
          color='r', size=20, ha='center', va='top')
axes.set_xlabel('Calinski-Harabaz index', fontsize=22)
axes.set_ylabel('Count', fontsize=22)
axes.set_title('Assay latent factor clustering\n', fontsize=22)
fig.show()
fig.savefig('../../predictd_figs/assay_lf_cluster_eval.pdf', bbox_inches='tight')


# In[15]:



