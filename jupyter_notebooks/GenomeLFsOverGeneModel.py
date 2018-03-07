
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
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy import stats
from sklearn.cluster import dbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import smart_open
import sparkpickle
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


### Load Genome Params

# In[3]:

gparams_key = 's3://predictd/PREDICTD/trained_predictd_models/test0/valid_fold0/genome_factors.rdd.pickle/part-{!s}.pickle'
num_parts = s3_library.get_pickle_s3(*s3_library.parse_s3_url(os.path.join(os.path.dirname(gparams_key), 'num_parts.pickle')))
gidx_list = []
gparams_list = []
for idx in xrange(num_parts):
    key_url = gparams_key.format(str(idx).zfill(5))
    with smart_open.smart_open(key_url, 'rb') as pickle_in:
        gidx, gparams = list(zip(*pickle.loads(pickle_in.read())))
        gidx_list.extend(gidx)
        g, gb = list(zip(*gparams))
        gparams_list.append(numpy.hstack([numpy.array(g),numpy.array(gb)[:,None]]))
gparams_array = numpy.vstack(gparams_list)
#only do latent factor clustering analysis on the latent factors; don't include the bias vector
gparams_genome_bias = gparams_array[:,-1]
gparams_array = gparams_array[:,:-1]
print('gparams_array.shape = {!s}'.format(gparams_array.shape))


# In[4]:

import random
rand_gparams_sample_idx = list(range(gparams_array.shape[0]))
random.shuffle(rand_gparams_sample_idx)
rand_gparams_sample = gparams_array[rand_gparams_sample_idx[:1000000]]
print(rand_gparams_sample.shape)
print(len(set(rand_gparams_sample_idx)))


# In[5]:

rand_gparams_sample_gb = gparams_genome_bias[rand_gparams_sample_idx[:1000000]]


# In[6]:

folds = s3_library.get_pickle_s3('predictd', 'reference_data/folds.5.8.pickle')
subsets = {'train':folds[0]['train'][0]['train'],
           'valid':folds[0]['train'][0]['valid'],
           'test':folds[0]['test']}


### Gene Model Take 2 (Normalized LF Values)

# In[7]:

gparams_array_pos = gparams_array #+ numpy.absolute(numpy.min(gparams_array, axis=0))
#print(numpy.sum(gparams_array_pos < 0))

#interquartile range outlier removal
pct75 = numpy.percentile(gparams_array_pos, 75, axis=0)
pct25 = numpy.percentile(gparams_array_pos, 25, axis=0)
iqr_val = 1.5 * (pct75 - pct25)
outlier_mask = numpy.logical_or(gparams_array_pos > (pct75 + iqr_val), gparams_array_pos < (pct25 - iqr_val))
#print(numpy.sum(outlier_mask)/numpy.multiply(*gparams_array_pos.shape, dtype=float))
iqr_masked = numpy.ma.masked_array(gparams_array_pos, mask=outlier_mask)
iqr_masked_min = iqr_masked - numpy.ma.min(iqr_masked, axis=0)
iqr_masked_norm = iqr_masked_min/numpy.ma.max(iqr_masked_min, axis=0)


# In[8]:

print(gidx_list[:5])


# In[9]:

print(numpy.min(iqr_masked_norm, axis=0))
print(numpy.max(iqr_masked_norm, axis=0))


# ##Compute Mean LF Values Across Gene Features

# In[10]:

gidx_dict = dict(zip(gidx_list, range(len(gidx_list))))


# In[11]:

'''I now have a bed file containing all gene parts binned into 10 bins per part for all primary transcripts of protein coding genes 
(the ones that have primary transcripts). This file was generated using the following command line in ./parse_gtf:

../parse_gtf_into_gene_parts.py ../gencode.v19.annotation.protein_coding.full.sorted.gtf.gz;
zcat gencode.v19.annotation.protein_coding.full.sorted.gene_model_windowed.bed.gz | sort -k1,1 -k2,2n | \
gzip > gencode.v19.annotation.protein_coding.full.sorted.gene_model_windowed.sorted.bed.gz;

Now I need to extract the corresponding imputation windows and map them to the bins on the different gene structures. I will use 
bedtools intersect -a <imputation_windows> -b <gene_model_windowed> -wao | cut -f 4,8,11
This will give me a tab-delimited file consisting of <imputation_window_name> <gene_model_window_category> <overlap_bp_count>
'''
#import subprocess
#cmd1 = ['bedtools', 'intersect', '-a', 'encodePilotRegions.hg19.sorted.windowed25bp.bed.gz', '-b', 
#        'parse_gtf/gencode.v19.annotation.protein_coding.full.sorted.gene_model_windowed.sorted.bed.gz',
#        '-wao']
#proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
##extract gidx, transcript/gene_category bin, overlap size
#cmd2 = ['cut', '-f', '1,2,7,10']
#proc2 = subprocess.Popen(cmd2, stdin=proc1.stdout, stdout=subprocess.PIPE)
#proc1.stdout.close()
#cmd3 = ['sort', '-k3,3']
#proc3 = subprocess.Popen(cmd3, stdin=proc2.stdout, stdout=subprocess.PIPE)
#proc2.stdout.close()
gene_category_dict = {}
gene_category_dict_win_idx = {}
overlap_dist = {}
prev_tx_id = None
tx_data = {}
win_path = 's3://predictd/reference_data/25bp_windows_overlapping_gene_model.txt.gz'
with smart_open.smart_open(win_path) as lines_in:
    for line in lines_in:
        genechr, genestart, genecat, overlap = line.strip().split()
        #just continue if the window does not overlap a gene category
        if genecat == '.':
            continue
        try:
            genewin_idx = gidx_dict[(genechr, int(genestart))]
        except KeyError:
            continue #continue if this coordinate is not found
        tx_id, genefeat, idx = genecat.split('-')
        if prev_tx_id and tx_id != prev_tx_id:
            for k in tx_data.keys():
                gene_category_dict.setdefault(k, []).append(numpy.mean(numpy.ma.vstack(tx_data[k]), axis=0))
            tx_data = {}
        tx_data.setdefault((genefeat, int(idx)), []).extend(int(overlap) * [iqr_masked_norm[genewin_idx]])
        prev_tx_id = tx_id
for genecat in list(gene_category_dict.keys()):
    gene_category_dict[genecat] = numpy.ma.vstack(gene_category_dict[genecat])


# In[12]:

#enhancer (p300) data
p300_win_dict = {}
p300_gidx_dict = {}
#import subprocess
#cmd1 = ['bedtools', 'intersect', '-a', 'encodePilotRegions.hg19.sorted.windowed25.bed.gz', '-b', 
#        'p300/all_p300_merged.windowed.bed.gz', '-wao']
#proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
##extract the rdd window name and the peak window name
#cmd2 = ['cut', '-f', '1,2,7,8']
#proc2 = subprocess.Popen(cmd2, stdin=proc1.stdout, stdout=subprocess.PIPE)
#proc1.stdout.close()
#cmd3 = ['sort', '-k3,3']
#proc3 = subprocess.Popen(cmd3, stdin=proc2.stdout, stdout=subprocess.PIPE)
#proc2.stdout.close()
prev_peak_name = None
peak_data = {}
win_path = 's3://predictd/reference_data/25bp_windows_overlapping_p300.txt.gz'
with smart_open.smart_open(win_path, 'rb') as data_in:
    for line in data_in:
        rdd_chr, rdd_start, peak_win, overlap = line.strip().split()
        if peak_win == '.':
            continue
        try:
            peak_win_gidx = gidx_dict[(rdd_chr, int(rdd_start))]
        except KeyError:
            continue
        #peaks are named as <p300_peak_name>-<peak_window_idx>
        peak_name, peak_win = peak_win.split('-')
        if prev_peak_name and peak_name != prev_peak_name:
            for k in peak_data.keys():
                p300_win_dict.setdefault(k, []).append(numpy.mean(numpy.ma.vstack(peak_data[k]), axis=0))
            peak_data = {}
        peak_data.setdefault(int(peak_win), []).extend(int(overlap) * [iqr_masked_norm[peak_win_gidx]])
        prev_peak_name = peak_name
for peakwin in list(p300_win_dict.keys()):
    p300_win_dict[peakwin] = numpy.ma.vstack(p300_win_dict[peakwin])


# In[13]:

print(p300_win_dict[4].shape)


# In[14]:

sorted_labels = ['promoter', 'utr_5prime', 'first_cds', 'first_intron', 'middle_cds', 'middle_intron', 'last_cds', 'last_intron', 
                 'utr_3prime', 'p300']
total_mat = []
for label in sorted_labels[:-1]:
    for idx in range(10):
        cat_mat = gene_category_dict[(label, idx)]
#        scale_lfs = (cat_mat + numpy.absolute(numpy.min(cat_mat, axis=0)))/numpy.max(cat_mat, axis=0)
        avg_per_lf = numpy.ma.mean(cat_mat, axis=0)
        total_mat.append(avg_per_lf[:, None])
for idx in range(10):
    cat_mat = p300_win_dict[idx]
    avg_per_lf = numpy.ma.mean(cat_mat, axis=0)
    total_mat.append(avg_per_lf[:,None])
total_mat = numpy.hstack(total_mat)
print(total_mat.shape)


# In[15]:

matplotlib.rcParams.update({'font.size': 22})
pyplot.rc('legend',**{'fontsize':20})
fig = pyplot.figure(figsize=(12,8)) 

#dendrogram plot for latent factors
ax1 = fig.add_axes([0, 0, 0.1, 0.9])
dist = distance.squareform(distance.pdist(total_mat, metric='euclidean'))
Y = hierarchy.linkage(dist, method='ward')
Z1 = hierarchy.dendrogram(Y, orientation='left', no_labels=True, ax=ax1)
ax1.set_xticks([])
ax1.set_xticklabels([])

axes = fig.add_axes([0.2,0,0.8,0.9])
#norm = matplotlib.colors.BoundaryNorm(pctl_norm, pyplot.cm.viridis.N)
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(total_mat), numpy.max(total_mat), 20), pyplot.cm.viridis.N)
#norm = None
im = axes.matshow(total_mat[Z1['leaves'],:], cmap='viridis', origin='lower', norm=norm, aspect='auto')
for val in numpy.arange(0,100,10) - 0.5:
    axes.axvline(val, color='black')
axes.set_xticks([])
axes.set_xlabel('Gene feature')
axes.set_ylabel('Genome latent factor')
fig.colorbar(im)

ax = fig.add_axes([0.2, 0.91, 0.64, 0.099])
#color_list = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5', '#d9d9d9',
#              '#bc80bd']
color_list = ['#8dd3c7','#ffffb3','#fdb462','#80b1d3','#fdb462','#80b1d3','#fdb462','#80b1d3','#ffffb3',
              '#bc80bd']
cmap = colors.ListedColormap(color_list)
bounds = [0,10,20,30,40,50,60,70,80,90,100]
norm = colors.BoundaryNorm(bounds, len(color_list))
feat_im = ax.matshow(numpy.arange(100)[None,:], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax.set_yticks([])
ax.set_xticks(numpy.arange(4.5,104.5,10))
ax.set_xticklabels(sorted_labels, rotation=45, ha='left')

##add conservation
#cons_ax = fig.add_axes([0.845, 0, 0.015, 0.9])
#im2 = cons_ax.matshow(cons_corr_vals[Z1['leaves'],None], cmap='bwr', origin='lower', aspect='auto')
#cons_ax.set_xticks([])
#cons_ax.set_yticks([])

fig.show()
fig.savefig('../../predictd_figs/genome_lf.pdf', bbox_inches='tight')
#data_dir = ('ADAM_BESTHYP/encodePilots_size-300_lf-69_rc-0.0288_ra-5.145e-13_ri-9.358e-14_'
#            'zero_bias_reg_init_uniform0.33_lrate0.005_batch_size_5k_pctl-weighted-err_'
#            'pctl-90_weight-40_pctl-99_weight-120_win-5kb')
#fig.savefig('tmp.pdf', bbox_inches='tight')
#key = s3_library.S3.get_bucket('encodeimputation2').new_key(os.path.join(data_dir, 'figs/genome_lfs_over_gene_model.pdf'))
#key.set_contents_from_filename('tmp.pdf')
#os.remove('tmp.pdf')


# In[16]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16,
                        'xtick.major.pad':8})

fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16,8), sharey=True) 

idx_lists = [list() for _ in range(2)]
lf_lists = [list() for _ in range(2)]
for i in xrange(total_mat.shape[0]):
    centered_lf_data = total_mat[i,90:] - total_mat[i,(90,-1)].mean()
    max_idx = numpy.where(numpy.absolute(centered_lf_data) == numpy.absolute(centered_lf_data).max())[0][0]
    if max_idx in [4,5]:
        idx_lists[0].append(centered_lf_data)
        lf_lists[0].append(i)
    else:
        idx_lists[1].append(centered_lf_data)
        lf_lists[1].append(i)
#    idx_lists[max_idx].append(centered_lf_data)
xlocs = numpy.arange(0,2000,200) - 900

axes[0].set_title('Max magnitude at P300 peak ({!s})\n'.format(len(idx_lists[0])), fontsize=20)
axes[0].set_xlabel('Base pairs around P300 peak', fontsize=20)
axes[0].set_ylabel('Centered latent factor magnitude', fontsize=20)
for lf_datum in idx_lists[0]:
    axes[0].plot(xlocs, lf_datum, linewidth=0.5)

axes[1].set_title('Max magnitude at P300 flank ({!s})\n'.format(len(idx_lists[1])), fontsize=20)
axes[1].set_xlabel('Base pairs around P300 peak', fontsize=20)
axes[1].set_ylabel('Centered latent factor magnitude', fontsize=20)
for lf_datum in idx_lists[1]:
    axes[1].plot(xlocs, lf_datum, linewidth=0.5)

fig.tight_layout()
fig.savefig('../../predictd_figs/genome_lf_p300.pdf', bbox_inches='tight')


# In[17]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16,
                        'xtick.major.pad':8})

fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16,8), sharey=True) 

idx_lists = [list() for _ in range(2)]
lf_lists = [list() for _ in range(2)]
for i in xrange(total_mat.shape[0]):
    centered_lf_data = total_mat[i,90:] - total_mat[i,(90,-1)].mean()
    max_idx = numpy.where(numpy.absolute(centered_lf_data) == numpy.absolute(centered_lf_data).max())[0][0]
    if max_idx in [4,5]:
        idx_lists[0].append(total_mat[i,90:])
        lf_lists[0].append(i)
    else:
        idx_lists[1].append(total_mat[i,90:])
        lf_lists[1].append(i)
#    idx_lists[max_idx].append(centered_lf_data)
xlocs = numpy.arange(0,2000,200) - 900

axes[0].set_title('Max magnitude at P300 peak ({!s})\n'.format(len(idx_lists[0])), fontsize=20)
axes[0].set_xlabel('Base pairs around P300 peak', fontsize=20)
axes[0].set_ylabel('Latent factor magnitude', fontsize=20)
for lf_datum in idx_lists[0]:
    axes[0].plot(xlocs, lf_datum, linewidth=0.5)

axes[1].set_title('Max magnitude at P300 flank ({!s})\n'.format(len(idx_lists[1])), fontsize=20)
axes[1].set_xlabel('Base pairs around P300 peak', fontsize=20)
axes[1].set_ylabel('Latent factor magnitude', fontsize=20)
for lf_datum in idx_lists[1]:
    axes[1].plot(xlocs, lf_datum, linewidth=0.5)

fig.tight_layout()


# In[18]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16,
                        'xtick.major.pad':8})

fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16,8), sharey=True) 

idx_lists = [list() for _ in range(2)]
lf_lists = [list() for _ in range(2)]
for i in xrange(total_mat.shape[0]):
    centered_lf_data = total_mat[i,90:] - total_mat[i,(90,-1)].mean()
    max_idx = numpy.where(numpy.absolute(centered_lf_data) == numpy.absolute(centered_lf_data).max())[0][0]
    if max_idx in [4,5]:
        idx_lists[0].append(total_mat[i,90:])
        lf_lists[0].append(i)
    else:
        idx_lists[1].append(total_mat[i,90:])
        lf_lists[1].append(i)

#sort the peaks and flanks latent factors by the range of values
peaks_sort_by_range = numpy.argsort(numpy.max(idx_lists[0], axis=1) - numpy.min(idx_lists[0], axis=1))[::-1]
flanks_sort_by_range = numpy.argsort(numpy.max(idx_lists[1], axis=1) - numpy.min(idx_lists[1], axis=1))[::-1]

xlocs = numpy.arange(0,2000,200) - 900

#Plot only the top ten latent factors of peaks and flanks by value range
axes[0].set_title('Max magnitude at P300 peak ({!s})\n'.format(len(idx_lists[0])), fontsize=20)
axes[0].set_xlabel('Base pairs around P300 peak', fontsize=20)
axes[0].set_ylabel('Latent factor magnitude', fontsize=20)
for idx in peaks_sort_by_range[:10]:
    lf_datum = idx_lists[0][idx]
    axes[0].plot(xlocs, lf_datum, linewidth=0.5)

axes[1].set_title('Max magnitude at P300 flank ({!s})\n'.format(len(idx_lists[1])), fontsize=20)
axes[1].set_xlabel('Base pairs around P300 peak', fontsize=20)
axes[1].set_ylabel('Latent factor magnitude', fontsize=20)
for idx in flanks_sort_by_range[:10]:
    lf_datum = idx_lists[1][idx]
    axes[1].plot(xlocs, lf_datum, linewidth=0.5)

fig.tight_layout()


# ###Check whether LFs recognizing P300 peaks also recognize H3K27ac, H3K4me1, etc.

# In[19]:

assay_params_url = 's3://predictd/PREDICTD/trained_predictd_models/test0/valid_fold0/assay_factors.pickle'
with smart_open.smart_open('s3://predictd/Observed/training_data.encodePilots_and_nchars.rdd.data_idx.pickle', 'rb') as pickle_in:
    data_idx = pickle.loads(pickle_in.read())
with smart_open.smart_open(assay_params_url, 'rb') as pickle_in:
    assay_params_tmp = pickle.loads(pickle_in.read())
#assay_params = numpy.hstack([assay_params_tmp[1][:,None], assay_params_tmp[0]])
assay_params = assay_params_tmp[0]

assay_labels = [assay[0] for assay in 
                 sorted(set([(elt[1],elt[-1][1]) for elt in data_idx.values()]),
                    key=lambda x:x[1])]
print('Num Assay Types: {!s}'.format(len(assay_labels)))
print('Shape of Assay Params: {!s}'.format(assay_params.shape))


# ###Incorporate cell type info

# In[20]:

ct_params_url = 's3://predictd/PREDICTD/trained_predictd_models/test0/valid_fold0/ct_factors.pickle'
with smart_open.smart_open(ct_params_url, 'rb') as pickle_in:
    ct_params_tmp = pickle.loads(pickle_in.read())
#ct_params = numpy.hstack([ct_params_tmp[1][:,None], ct_params_tmp[0]])
ct_params = ct_params_tmp[0]

ct_labels = [ct[0].replace('_', ' ') for ct in 
             sorted(set([(elt[0],elt[-1][0]) for elt in data_idx.values()]),
                    key=lambda x:x[1])]
print('Num Cell Types: {!s}'.format(len(ct_labels)))
print('Shape of Cell Type Params: {!s}'.format(ct_params.shape))


# In[21]:

#just pure assay latent factor values
peak_lfs_assay_values = assay_params[:, lf_lists[0]]
flank_lfs_assay_values = assay_params[:, lf_lists[1]]

from sklearn.metrics import roc_curve
fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8,8))
axes.plot((0,1), (0,1), color='k', linestyle=':')
for label, color in [('H3K4me1', 'blue'), ('H3K27ac', 'red'), ('DNase', 'brown'), ('H3K4me3', 'green'),
                     ('H3K27me3', 'pink'), ('H3K9me3', 'orange'), ('H3K36me3', 'purple')]:
    label_idx = assay_labels.index(label)
    peak_vals = peak_lfs_assay_values[label_idx].flatten()
    flank_vals = flank_lfs_assay_values[label_idx].flatten()
    fpr, tpr, thresholds = roc_curve(numpy.array([1] * peak_vals.shape[0] + [0] * flank_vals.shape[0]),
                                     numpy.concatenate((peak_vals, flank_vals)))
    axes.plot(fpr, tpr, color=color, label=label, linewidth=3)

#axes.set_title('Are H3K4me1 Assay Values Higher for "Peak" Genome Latent Factors')
axes.legend(bbox_to_anchor=(1.43, 1.0))
axes.set_ylabel('TPR (peaks)')
axes.set_xlabel('FPR (flanks)')

fig.tight_layout()


# In[22]:

#ct x assay values summed across latent factors
ct_assay_final_sign = numpy.dstack([ct_params[i] * assay_params for i in range(ct_params.shape[0])]).swapaxes(1,2) * numpy.sign(total_mat[:,94])
print(ct_assay_final_sign.shape)

from sklearn.metrics import roc_curve
fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8,8))
axes.plot((0,1), (0,1), color='k', linestyle=':')
for label, color in [('H3K4me1', 'blue'), ('H3K27ac', 'red'), ('DNase', 'brown'), ('H3K4me3', 'green'),
                     ('H3K27me3', 'pink'), ('H3K9me3', 'orange'), ('H3K36me3', 'purple')]:
    label_idx = assay_labels.index(label)
    
    peak_vals = numpy.sum(ct_assay_final_sign[label_idx].squeeze()[:,lf_lists[0]], axis=1)
    flank_vals = numpy.sum(ct_assay_final_sign[label_idx].squeeze()[:,lf_lists[1]], axis=1)
    group_idx = numpy.array([1] * peak_vals.shape[0] + [0] * flank_vals.shape[0])
    fpr, tpr, threshold = roc_curve(group_idx, numpy.concatenate((peak_vals, flank_vals)))
    axes.plot(fpr, tpr, color=color, label=label, linewidth=3)

#axes.set_title('Are H3K4me1 Assay Values Higher for "Peak" Genome Latent Factors')
axes.legend(bbox_to_anchor=(1.43, 1.0))
axes.set_ylabel('TPR (peaks)')
axes.set_xlabel('FPR (flanks)')

fig.tight_layout()


# In[23]:

rand_gparams_sample_imp = []
gmean = s3_library.get_pickle_s3('predictd', 'PREDICTD/trained_predictd_models/test0/valid_fold0/gmean.pickle')
for gidx in range(min(rand_gparams_sample.shape[0], 100000)):
    impvals = (numpy.sum([numpy.outer(ct_params[:, lfidx], assay_params[:, lfidx]) * rand_gparams_sample[gidx, lfidx] 
                         for lfidx in range(1,rand_gparams_sample.shape[1])], axis=0)
               + rand_gparams_sample_gb[gidx] + ct_params_tmp[1][:,None] + assay_params_tmp[1])[~subsets['test']] + gmean
    rand_gparams_sample_imp.append(impvals)
rand_gparams_sample_imp = numpy.array(rand_gparams_sample_imp)


# In[24]:

fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8,8))
histvals = axes.hist(rand_gparams_sample_imp.flatten(), bins=1000)
axes.axvline(0, color='r')
axes.set_xlabel('Imputed value')
axes.set_ylabel('Count')
fig.tight_layout()
fig.savefig('../../predictd_figs/imputed_value_histogram.pdf', bbox_inches='tight')


# ##Latent Factor Correlation Plots

# In[25]:

def pairwise_column_corr(to_corr):
    import itertools
    from scipy.stats import pearsonr, spearmanr
    corrs = numpy.zeros((to_corr.shape[1], to_corr.shape[1]))
    for idx1, idx2 in itertools.product(numpy.arange(to_corr.shape[1]), numpy.arange(to_corr.shape[1])):
        corr_val = pearsonr(to_corr[:,idx1], to_corr[:,idx2])
        corrs[idx1, idx2] = corr_val[0]
        corrs[idx2, idx1] = corr_val[0]
    return corrs

#to_corr = numpy.vstack([rand_gparams_sample[:10000], ct_params, assay_params])
genome_corr = pairwise_column_corr(rand_gparams_sample)
ct_corr = pairwise_column_corr(ct_params)
assay_corr = pairwise_column_corr(assay_params)


# In[26]:

print('Mean genome_corr: {!s}\nMean ct_corr: {!s}\nMean assay_corr: {!s}'.
      format(numpy.mean(genome_corr), numpy.mean(ct_corr), numpy.mean(assay_corr)))


# In[27]:

pyplot.rcParams.update({'xtick.labelsize':18,
                        'ytick.labelsize':18})

for title, corrs in [('Genome LF Pearson', genome_corr),
                     ('CT LF Pearson', ct_corr),
                     ('Assay LF Pearson', assay_corr)]:
    fig = pyplot.figure(figsize=(10,8))
    #dendrogram plot for lfs
    ax2 = fig.add_axes([0, 0.975, 0.73, 0.05])
    Y = hierarchy.linkage(corrs, method='average')
    Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, ax=ax2)
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    #LF matrix plot
    ax3 = fig.add_axes([0, 0.02, 0.73, 0.95])
    x_index = Z2['leaves']
    y_index = Z2['leaves']
    resort_corrs = corrs[y_index,:]
    resort_corrs = resort_corrs[:,x_index]
    im = ax3.matshow(resort_corrs, cmap='viridis', origin='lower')
    ax3.set_xticks([])

    #left, bottom, width, height
    #cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
    cbaxes = fig.add_axes([0, 0, 0.73, 0.015]) 
    cb = fig.colorbar(im, orientation='horizontal', cax=cbaxes)
    fig.show()


# In[28]:

total_corr = pairwise_column_corr(numpy.vstack([ct_params, assay_params, rand_gparams_sample]))


# In[29]:

print(total_corr.shape)
print(numpy.mean(numpy.absolute(total_corr)))


# In[30]:

fig = pyplot.figure(figsize=(8,8))
#dendrogram plot for lfs
ax2 = fig.add_axes([0, 0.975, 0.95, 0.05])
Y = hierarchy.linkage(total_corr, method='average')
Z2 = hierarchy.dendrogram(Y, orientation='top', no_labels=True, ax=ax2)
ax2.set_yticks([])
ax2.set_yticklabels([])

#LF matrix plot
ax3 = fig.add_axes([0, 0, 0.95, 0.95])
x_index = Z2['leaves']
y_index = Z2['leaves']
resort_corrs = total_corr[y_index,:]
resort_corrs = resort_corrs[:,x_index]

#norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(total_corr), numpy.max(total_corr), 20), pyplot.cm.bwr.N)
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(-1.0, 1.0, 16), pyplot.cm.bwr.N)
im = ax3.matshow(resort_corrs, cmap='bwr', norm=norm, origin='lower')
#ax3.set_xticks([])
ax3.set_ylabel('Latent factors', fontsize=18)
ax3.set_xlabel('Latent factors', fontsize=18)
ax3.xaxis.tick_bottom()

#left, bottom, width, height
#cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
#cbaxes = fig.add_axes([0, 0, 0.73, 0.015])
cbaxes = fig.add_axes([0.985, 0.225, 0.015, 0.5])
cbar = fig.colorbar(im, orientation='vertical', cax=cbaxes)
cbar.set_ticks([-1.0, -0.5, 0, 0.5, 1.0])
cbar.set_label('Pearson correlation', fontsize=16)
#cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='vertical')
fig.show()


# In[31]:

fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(4,8))
axes.violinplot(numpy.triu(total_corr, k=1).flatten())
axes.set_ylabel('Pairwise pearson correlation')
axes.set_xticklabels([])
axes.set_ylim((-1.0, 1.0))
fig.show()


# ##Permute Latent Factors to Find Null

# In[32]:

print(iqr_masked_norm.shape)


# In[33]:

'''I now have a bed file containing all gene parts binned into 10 bins per part for all primary transcripts of protein coding genes 
(the ones that have primary transcripts). This file was generated using the following command line in ./parse_gtf:

../parse_gtf_into_gene_parts.py ../gencode.v19.annotation.protein_coding.full.sorted.gtf.gz;
zcat gencode.v19.annotation.protein_coding.full.sorted.gene_model_windowed.bed.gz | sort -k1,1 -k2,2n | \
gzip > gencode.v19.annotation.protein_coding.full.sorted.gene_model_windowed.sorted.bed.gz;

Now I need to extract the corresponding imputation windows and map them to the bins on the different gene structures. I will use 
bedtools intersect -a <imputation_windows> -b <gene_model_windowed> -wao | cut -f 4,8,11
This will give me a tab-delimited file consisting of <imputation_window_name> <gene_model_window_category> <overlap_bp_count>
'''
#import subprocess
#cmd1 = ['bedtools', 'intersect', '-a', 'encodePilotRegions.hg19.sorted.windowed.bed', '-b', 
#        'parse_gtf/gencode.v19.annotation.protein_coding.full.sorted.gene_model_windowed.sorted.bed.gz',
#        '-wao']
#proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
##extract gidx, transcript/gene_category bin, overlap size
#cmd2 = ['cut', '-f', '4,8,11']
#proc2 = subprocess.Popen(cmd2, stdin=proc1.stdout, stdout=subprocess.PIPE)
#proc1.stdout.close()
#cmd3 = ['sort', '-k1,1', '-k2,2n']
#proc3 = subprocess.Popen(cmd3, stdin=proc2.stdout, stdout=subprocess.PIPE)
#proc2.stdout.close()
gene_category_dict = {}
gene_category_dict_win_idx = {}
overlap_dist = {}
prev_tx_id = None
tx_data = {}
num_skipped = 0
#with proc3.stdout as lines_in:
win_path = 's3://predictd/reference_data/25bp_windows_overlapping_gene_model.txt.gz'
with smart_open.smart_open(win_path, 'rb') as lines_in:
    for line in lines_in:
        genechr, genestart, genecat, overlap = line.strip().split()
        #just continue if the window does not overlap a gene category
        if genecat == '.':
            continue
        try:
            genewin_idx = gidx_dict[(genechr, int(genestart))]
        except KeyError:
            num_skipped += 1
            continue
        tx_id, genefeat, idx = genecat.split('-')
        if prev_tx_id and tx_id != prev_tx_id:
            for k in tx_data.keys():
                gene_category_dict.setdefault(k, []).append(numpy.mean(numpy.ma.vstack(tx_data[k]), axis=0))
            tx_data = {}
        tx_data.setdefault((genefeat, int(idx)), []).extend(int(overlap) * [iqr_masked_norm[genewin_idx, numpy.random.permutation(iqr_masked_norm.shape[1])]])
        prev_tx_id = tx_id
for genecat in list(gene_category_dict.keys()):
    gene_category_dict[genecat] = numpy.ma.vstack(gene_category_dict[genecat])


# In[34]:

print(num_skipped)


# In[35]:

#enhancer (p300) data
p300_win_dict = {}
p300_gidx_dict = {}
#import subprocess
#cmd1 = ['bedtools', 'intersect', '-a', 'encodePilotRegions.hg19.sorted.windowed.bed', '-b', 
#        'p300/all_p300_merged.windowed.bed.gz', '-wao']
#proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
##extract the rdd window name and the peak window name
#cmd2 = ['cut', '-f', '4,8,9']
#proc2 = subprocess.Popen(cmd2, stdin=proc1.stdout, stdout=subprocess.PIPE)
#proc1.stdout.close()
prev_peak_name = None
peak_data = {}
#with proc2.stdout as data_in:
win_path = 's3://predictd/reference_data/25bp_windows_overlapping_p300.txt.gz'
with smart_open.smart_open(win_path, 'rb') as data_in:
    for line in data_in:
        rdd_chr, rdd_start, peak_win, overlap = line.strip().split()
        if peak_win == '.':
            continue
        try:
            peak_win_gidx = gidx_dict[(rdd_chr, int(rdd_start))]
        except KeyError:
            continue
        #peaks are named as <p300_peak_name>-<peak_window_idx>
        peak_name, peak_win = peak_win.split('-')
        if prev_peak_name and peak_name != prev_peak_name:
            for k in peak_data.keys():
                p300_win_dict.setdefault(k, []).append(numpy.mean(numpy.ma.vstack(peak_data[k]), axis=0))
            peak_data = {}
        peak_data.setdefault(int(peak_win), []).extend(int(overlap) * [iqr_masked_norm[peak_win_gidx, numpy.random.permutation(iqr_masked_norm.shape[1])]])
        prev_peak_name = peak_name
for peakwin in list(p300_win_dict.keys()):
    p300_win_dict[peakwin] = numpy.ma.vstack(p300_win_dict[peakwin])


# In[36]:

sorted_labels = ['promoter', 'utr_5prime', 'first_cds', 'first_intron', 'middle_cds', 'middle_intron', 'last_cds', 'last_intron', 
                 'utr_3prime', 'p300']
total_mat = []
for label in sorted_labels[:-1]:
    for idx in range(10):
        cat_mat = gene_category_dict[(label, idx)]
#        scale_lfs = (cat_mat + numpy.absolute(numpy.min(cat_mat, axis=0)))/numpy.max(cat_mat, axis=0)
        avg_per_lf = numpy.ma.mean(cat_mat, axis=0)
        total_mat.append(avg_per_lf[:, None])
for idx in range(10):
    cat_mat = p300_win_dict[idx]
    avg_per_lf = numpy.ma.mean(cat_mat, axis=0)
    total_mat.append(avg_per_lf[:,None])
total_mat = numpy.hstack(total_mat)
print(total_mat.shape)


# In[37]:

matplotlib.rcParams.update({'font.size': 22})
pyplot.rc('legend',**{'fontsize':20})
fig = pyplot.figure(figsize=(12,8)) 

#dendrogram plot for latent factors
ax1 = fig.add_axes([0, 0, 0.1, 0.9])
dist = distance.squareform(distance.pdist(total_mat, metric='euclidean'))
Y = hierarchy.linkage(dist, method='ward')
Z1 = hierarchy.dendrogram(Y, orientation='left', no_labels=True, ax=ax1)
ax1.set_xticks([])
ax1.set_xticklabels([])

axes = fig.add_axes([0.2,0,0.8,0.9])
#norm = matplotlib.colors.BoundaryNorm(pctl_norm, pyplot.cm.viridis.N)
norm = matplotlib.colors.BoundaryNorm(numpy.linspace(numpy.min(total_mat), numpy.max(total_mat), 20), pyplot.cm.viridis.N)
#norm = None
im = axes.matshow(total_mat[Z1['leaves'],:], cmap='viridis', origin='lower', norm=norm, aspect='auto')
for val in numpy.arange(0,100,10) - 0.5:
    axes.axvline(val, color='black')
axes.set_xticks([])
axes.set_xlabel('Gene feature')
axes.set_ylabel('Genome latent factor')
fig.colorbar(im)

ax = fig.add_axes([0.2, 0.91, 0.64, 0.099])
#color_list = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5', '#d9d9d9',
#              '#bc80bd']
color_list = ['#8dd3c7','#ffffb3','#fdb462','#80b1d3','#fdb462','#80b1d3','#fdb462','#80b1d3','#ffffb3',
              '#bc80bd']
cmap = colors.ListedColormap(color_list)
bounds = [0,10,20,30,40,50,60,70,80,90,100]
norm = colors.BoundaryNorm(bounds, len(color_list))
feat_im = ax.matshow(numpy.arange(100)[None,:], cmap=cmap, norm=norm, origin='lower', aspect='auto')
ax.set_yticks([])
ax.set_xticks(numpy.arange(4.5,104.5,10))
ax.set_xticklabels(sorted_labels, rotation=45, ha='left')

##add conservation
#cons_ax = fig.add_axes([0.845, 0, 0.015, 0.9])
#im2 = cons_ax.matshow(cons_corr_vals[Z1['leaves'],None], cmap='bwr', origin='lower', aspect='auto')
#cons_ax.set_xticks([])
#cons_ax.set_yticks([])

fig.show()
fig.savefig('../../predictd_figs/genome_lf_random.pdf', bbox_inches='tight')
#data_dir = ('ADAM_BESTHYP/encodePilots_size-300_lf-69_rc-0.0288_ra-5.145e-13_ri-9.358e-14_'
#            'zero_bias_reg_init_uniform0.33_lrate0.005_batch_size_5k_pctl-weighted-err_'
#            'pctl-90_weight-40_pctl-99_weight-120_win-5kb')
#fig.savefig('tmp.pdf', bbox_inches='tight')
#key = s3_library.S3.get_bucket('encodeimputation2').new_key(os.path.join(data_dir, 'figs/genome_lfs_over_gene_model.pdf'))
#key.set_contents_from_filename('tmp.pdf')
#os.remove('tmp.pdf')


# In[37]:



