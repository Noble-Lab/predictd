
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'png'")

import copy
import glob
import gzip
import matplotlib
from matplotlib import pyplot
import numpy
import os
import pickle
import random
from scipy.spatial import distance
from scipy.cluster import hierarchy
import struct
import sys

sys.path.append('..')
import s3_library


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


### Get DataIdx and Create Matrix

# In[3]:

data_idx = s3_library.get_pickle_s3('predictd', 'Observed/training_data.encodePilots_and_nchars.rdd.data_idx.pickle')
assay_labels = [assay[0].replace('H2AZ', 'H2A.Z') for assay in sorted(set([(elt[1], elt[-1][1]) for elt in data_idx.values()]), key=lambda x:x[1])]
ct_labels = [ct[0].replace('_', ' ') for ct in sorted(set([(elt[0],elt[-1][0]) for elt in data_idx.values()]), key=lambda x:x[1])]
mat = numpy.zeros((127, 24))
mat[list(zip(*[elt[-1] for elt in data_idx.values()]))] = 1


# In[4]:

#sort cell types and assays by the number of available experiments
#and induce this order on the matrix rows and columns
ct_index = numpy.argsort(numpy.sum(mat, axis=1))
assay_index = numpy.argsort(numpy.sum(mat, axis=0))[::-1]
resort_mat = mat[ct_index,:]
resort_mat = resort_mat[:,assay_index]

#plot the matrix
fig = pyplot.figure(figsize=(20,30))
ax = fig.add_axes([0.12, 0.01, 0.5, 0.75])
im = ax.matshow(resort_mat, cmap='viridis', origin='lower', aspect='auto', extent=(0, 24, 0, 127))

#label and format the y axis
ax.set_yticks(numpy.arange(mat.shape[0]))
ax.set_yticklabels(numpy.array(ct_labels)[ct_index], va='bottom', fontsize=14)

#label and format the x axis
ax.set_xticks(numpy.arange(mat.shape[1]))
ax.set_xticklabels(numpy.array(assay_labels)[assay_index], rotation=45, ha='left', fontsize=18)

#show the grid on the heatmap
ax.grid()

#show the figure
fig.show()
fig.savefig('../../predictd_figs/imputation_data_matrix.pdf', bbox_inches='tight')

