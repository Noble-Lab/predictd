
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import itertools
import matplotlib
from matplotlib import pyplot
import numpy
import os
import pickle
import shlex
import smart_open
import sparkpickle
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


# ##Code for loading hyperparameter results

# In[3]:

#iter_errs_minvals column order is as follows
#0       1            2          3       4       5       6        7                     8                    9                       10
#lf_num, sp_iter_num, lrate_val, ra_val, rc_val, ri_val, ri2_val, final_valid_err_iter, final_valid_err_val, min_sgd_valid_err_iter, min_sgd_valid_err_val,
#11        12
#fold_idx, valid_fold_idx
iter_errs_minvals = s3_library.get_pickle_s3('predictd', 'PREDICTD/hyperparameter_search/20180223_iter_errs_minvals_data.pickle')


# In[4]:

'''
Note: This is the code that was used to read in the hyperparameter search results. These S3 locations no longer exist, so the data is now read in 
from a pickle file in the cell above. The Spearmint configuration files, databases, and output files can be found in the S3 location 
specified in the code repository for the PREDICTD paper.


iter_errs_keys = s3_library.glob_keys('hypersearch', 'hyperparam_search/lf*/*/ct_factors.pickle')
iter_errs_keys += s3_library.glob_keys('hypersearch', 'gtotal_valid_test/lf*/*/ct_factors.pickle')
iter_errs_keys += s3_library.glob_keys('hypersearch', 'for_pub/lf100/*/ct_factors.pickle')
bucket = s3_library.S3.get_bucket('hypersearch')
iter_errs_keys_table = sorted([tuple(elt.name.strip().split('/')[:3] + 
                                     [bucket.get_key(elt.name.replace('ct_factors.pickle', 'iter_errs.txt'))]) 
                               for elt in iter_errs_keys])
existing_iters = set([tuple(elt[:2]) for elt in iter_errs_minvals])
for expt_type, lf_num, iter_num, iter_errs_key in iter_errs_keys_table:
    lf_num = float(lf_num[2:])
    if expt_type == 'gtotal_valid_test':
        lf_num += 0.3
    elif expt_type == 'for_pub':
        lf_num = 101
    iter_num = int(iter_num.split('_')[1])
    #get iter_errs, skip header line
    iter_errs = [elt.split() for elt in iter_errs_key.get_contents_as_string().strip().split('\n')][1:]
    min_sgd_valid_err = iter_errs[numpy.argmin(numpy.array(iter_errs, dtype=object)[:-1,3].flatten().astype(float))]
    final_valid_err = iter_errs[-1]

    #get hyperparameter values
    command_line_key = s3_library.S3.get_bucket('hypersearch').get_key(iter_errs_key.name.replace('iter_errs', 'command_line'))
    if command_line_key is None:
        continue
    parsed_cmd = shlex.split(command_line_key.get_contents_as_string())
    optimized_hyperparams = ['--learning_rate', '--ra', '--rc', '--ri', '--ri2', '--fold_idx', '--valid_fold_idx']
    hyp_vals = {k:float(v) for k,v in [elt.lstrip('-').split('=') for elt in parsed_cmd if elt.split('=')[0] in optimized_hyperparams]}

    #record everything together
    #0       1            2          3       4       5       6        7                     8                    9                       10
    #lf_num, sp_iter_num, lrate_val, ra_val, rc_val, ri_val, ri2_val, final_valid_err_iter, final_valid_err_val, min_sgd_valid_err_iter, min_sgd_valid_err_val,
    #11        12
    #fold_idx, valid_fold_idx
    if (lf_num, iter_num) not in existing_iters:
        iter_errs_minvals.append((lf_num, iter_num, hyp_vals['learning_rate'], hyp_vals['ra'], hyp_vals['rc'], 
                                  hyp_vals['ri'], hyp_vals['ri2'], final_valid_err[0], final_valid_err[3], min_sgd_valid_err[0], min_sgd_valid_err[3],
                                  hyp_vals['fold_idx'], hyp_vals['valid_fold_idx']))
iter_errs_minvals.sort()
'''


# ##Plot Minimum Validation MSE Encountered by Spearmint

# In[5]:

min_iters_threshold = 50
min_iters_pct = 0.4
min_gap_threshold = 20
min_gap_pct = 0.15

iter_errs_array = numpy.array(iter_errs_minvals, dtype=float)
lf_nums = [int(elt) for elt in sorted(set(iter_errs_array[:,0].astype(float))) if elt == round(elt) and elt != 101]
min_iters = []
min_idxs = []
iter_thresholds = []
for lf_num in lf_nums:
    lf_num_total_rows = iter_errs_array[numpy.where(iter_errs_array[:,0] == lf_num)[0]]
    #ignore stopping criterion for the publication hyperparameter search
    if lf_num == 101:
        lf_num_rows = lf_num_total_rows
        min_val = numpy.min(lf_num_rows[numpy.where(lf_num_rows[:,8] < lf_num_rows[:,10])[0],8])
        min_idx = numpy.where(lf_num_rows[:,8] == min_val)[0][0]
        iter_thresholds.append(lf_num_rows.shape[0])
        min_idxs.append(min_idx)
        min_iters.append(tuple(lf_num_rows[min_idx]))
        continue
    elif (min_iters_pct * lf_num) < min_iters_threshold:
        lf_num_rows = lf_num_total_rows[:min_iters_threshold]
    else:
        lf_num_rows = lf_num_total_rows[:int(numpy.ceil(min_iters_pct * lf_num))]
    not_passing = True
    message = ''
    while not_passing:
        min_val = numpy.min(lf_num_rows[numpy.where(lf_num_rows[:,8] < lf_num_rows[:,10])[0],8])
        min_idx = numpy.where(lf_num_rows[:,8] == min_val)[0][0]
        gap = lf_num_rows.shape[0] - (min_idx + 1)
        target_gap = numpy.ceil(min_gap_pct * lf_num)
        if (lf_num_rows.shape[0] < min_iters_threshold) or (lf_num_rows.shape[0] < numpy.ceil(min_iters_pct * lf_num)):
            message = 'Experiment for {!s} latent factors has not progressed far enough.'.format(lf_num)
        elif gap < min_gap_threshold:
            message = 'Experiment for {!s} latent factors has not gone {!s} iterations since the last minimum.'.format(lf_num, min_gap_threshold) 
        elif gap < target_gap:
            message = 'Experiment for {!s} latent factors has a gap of {!s}, not {!s}.'.format(lf_num, gap, target_gap)
        else:
            not_passing = False
            message = ''
            break
        if lf_num_rows.shape[0] == lf_num_total_rows.shape[0]:
            break
        lf_num_rows = lf_num_total_rows[:lf_num_rows.shape[0] + 1]
    if not_passing is True:
        print(message)
    #save the number of iterations at which the stopping criterion is met
    iter_thresholds.append(lf_num_rows.shape[0])
    #save the iteration that has the minimum value of these iterations
    min_idxs.append(min_idx)
    #compile the data for the minimum iterations
    min_iters.append(tuple(lf_num_rows[min_idx]))
min_iters = numpy.array(min_iters)

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16,
                        'xtick.major.pad':8})
fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16,8))
ind = numpy.arange(1, len(lf_nums) * 2, 2)
width = 1.0
axes[0].bar(ind, min_iters[:,8].flatten(), width, align='edge')
axes[0].set_ylabel('Minimum validation MSE', fontsize=20)
axes[0].set_xlabel('Number of latent factors', fontsize=20)
axes[0].set_xticks(ind + ((width)/2))
axes[0].set_xticklabels(lf_nums, rotation=90)

#num_iters = [len(numpy.where(iter_errs_array[:,0] == lf_num)[0]) for lf_num in lf_nums]
num_iters = iter_thresholds
#ax2 = axes.twinx()
ax2 = axes[1]
ax2.bar(ind, num_iters, width, align='edge', color='red', label='Total Spearmint iterations')
ax2.bar(ind, min_idxs, width, align='edge', color='green', label='Iterations to minimum MSE')
ax2.set_ylabel('Number of Spearmint iterations', fontsize=20)
ax2.set_xlabel('Number of latent factors', fontsize=20)
ax2.legend(loc='best')
ax2.set_xticks(ind + ((width)/2))
ax2.set_xticklabels(lf_nums, rotation=90)

fig.tight_layout()


# ###Get extra Spearmint iteration info for 32 and 64 latent factors

# In[6]:

min_iters_threshold = 120
min_iters_pct = 0
min_gap_threshold = 1
min_gap_pct = 0

iter_errs_array = numpy.array(iter_errs_minvals, dtype=float)
lf_nums = [int(elt) for elt in sorted(set(iter_errs_array[:,0].astype(float))) if elt == round(elt) and elt != 101]
min_iters_all_spearmint = []
min_idxs_all_spearmint = []
iter_thresholds = []
for lf_num in lf_nums:
    lf_num_total_rows = iter_errs_array[numpy.where(iter_errs_array[:,0] == lf_num)[0]]
    #ignore stopping criterion for the publication hyperparameter search
    if lf_num == 101:
        lf_num_rows = lf_num_total_rows
        min_val = numpy.min(lf_num_rows[numpy.where(lf_num_rows[:,8] < lf_num_rows[:,10])[0],8])
        min_idx = numpy.where(lf_num_rows[:,8] == min_val)[0][0]
        iter_thresholds.append(lf_num_rows.shape[0])
        min_idxs_all_spearmint.append(min_idx)
        min_iters_all_spearmint.append(tuple(lf_num_rows[min_idx]))
        continue
    elif (lf_num in [32, 64]) or (min_iters_pct * lf_num) < min_iters_threshold:
        lf_num_rows = lf_num_total_rows[:min_iters_threshold]
    else:
        lf_num_rows = lf_num_total_rows[:int(numpy.ceil(min_iters_pct * lf_num))]
    not_passing = True
    message = ''
    while not_passing:
        min_val = numpy.min(lf_num_rows[numpy.where(lf_num_rows[:,8] < lf_num_rows[:,10])[0],8])
        min_idx = numpy.where(lf_num_rows[:,8] == min_val)[0][0]
        gap = lf_num_rows.shape[0] - (min_idx + 1)
        target_gap = numpy.ceil(min_gap_pct * lf_num)
        if (lf_num_rows.shape[0] < min_iters_threshold) or (lf_num_rows.shape[0] < numpy.ceil(min_iters_pct * lf_num)):
            message = 'Experiment for {!s} latent factors has not progressed far enough.'.format(lf_num)
        elif gap < min_gap_threshold:
            message = 'Experiment for {!s} latent factors has not gone {!s} iterations since the last minimum.'.format(lf_num, min_gap_threshold) 
        elif gap < target_gap:
            message = 'Experiment for {!s} latent factors has a gap of {!s}, not {!s}.'.format(lf_num, gap, target_gap)
        else:
            not_passing = False
            message = ''
            break
        if lf_num_rows.shape[0] == lf_num_total_rows.shape[0]:
            break
        lf_num_rows = lf_num_total_rows[:lf_num_rows.shape[0] + 1]
    if not_passing is True:
        print(message)
    #save the number of iterations at which the stopping criterion is met
    iter_thresholds.append(lf_num_rows.shape[0])
    #save the iteration that has the minimum value of these iterations
    min_idxs_all_spearmint.append(min_idx)
    #compile the data for the minimum iterations
    min_iters_all_spearmint.append(tuple(lf_num_rows[min_idx]))
    if lf_num in [32, 64]:
        min_candidates = numpy.where(lf_num_rows[:,8] < lf_num_rows[:,10])[0]
        min_candidates_order = numpy.argsort(lf_num_rows[min_candidates,8])
        print(lf_num, min_candidates[min_candidates_order][:10] + 1)
    if lf_num == 32:
        lf_num_32_next_min = lf_num_rows[min_candidates[min_candidates_order[3]]][8]
    if lf_num == 64:
        lf_num_64_next_min = lf_num_rows[min_candidates[min_candidates_order[2]]][8]
min_iters_all_spearmint = numpy.array(min_iters_all_spearmint)
num_iters_all_spearmint = iter_thresholds


# ###Plot everything together

# In[7]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16,
                        'xtick.major.pad':8})
fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16,8))
ind = numpy.arange(1, len(lf_nums) * 2, 2)
width = 0.5
axes[0].bar(ind, min_iters[:,8].flatten(), width, align='edge', label='Thresholded Spearmint search')
axes[0].set_ylabel('Minimum validation MSE', fontsize=20)
axes[0].set_xlabel('Number of latent factors', fontsize=20)
axes[0].set_xticks(ind + ((width)/2))
axes[0].set_xticklabels(lf_nums, rotation=90)

axes[0].bar(ind[5:7] + width, min_iters_all_spearmint[5:7,8].flatten(), width, align='edge', color='brown', label='Expanded Spearmint Search')
axes[0].legend(loc='best')

ax2 = axes[1]
ax2.bar(ind, num_iters, width, align='edge', color='red', label='Total Spearmint iterations')
ax2.bar(ind, min_idxs, width, align='edge', color='blue', label='Iterations to minimum MSE')
ax2.bar(ind[5:7] + width, num_iters_all_spearmint[5:7], width, align='edge', color='orange', label='Expanded total Spearmint iterations')
ax2.bar(ind[5:7] + width, min_idxs_all_spearmint[5:7], width, align='edge', color='brown', label='Expanded iterations to minimum MSE')
ax2.set_ylabel('Number of Spearmint iterations', fontsize=20)
ax2.set_xlabel('Number of latent factors', fontsize=20)
ax2.legend(loc='best')
ax2.set_xticks(ind + ((width)/2))
ax2.set_xticklabels(lf_nums, rotation=90)

fig.tight_layout()
fig.savefig('../../predictd_figs/hypersearch.pdf', bbox_inches='tight')


# ###Plot Distributions of best results of the ten trained models per number of latent factors

# In[8]:

#get the best results
all_mse_vals = []
for lf_num in [2,4,8,16,24,32,64,100,128,153,178,203,228,256,341,426,512]:
    if lf_num in [32, 64]:
        key_glob = ('PREDICTD/hyperparameter_search/test_hypersearch_results_on_fixed_validation_set_and_fixed_genomic_windows/'
                    '{!s}_no_threshold/?/iter_errs.txt')
    else:
        key_glob = ('PREDICTD/hyperparameter_search/test_hypersearch_results_on_fixed_validation_set_and_fixed_genomic_windows/'
                    '{!s}/?/iter_errs.txt')
    model_keys = []
    mse_vals = []
    for key in s3_library.glob_keys('predictd', key_glob.format(lf_num)):
        model_keys.append(key.name)
        with smart_open.smart_open(key) as lines_in:
            mse_vals.append(float(lines_in.read().strip().split('\n')[-1].split()[3]))
    all_mse_vals.append(mse_vals)


# In[9]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16,
                        'xtick.major.pad':8})
fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8,8))
ind = numpy.arange(1, len(all_mse_vals) * 2, 2)
axes.violinplot(all_mse_vals, ind, points=60, widths=0.5, showmeans=True,
                showextrema=True, showmedians=False, bw_method=0.5)
axes.set_ylabel('Validation MSE', fontsize=20)
axes.set_xlabel('Number of latent factors', fontsize=20)
axes.set_xticks(ind)
axes.set_xticklabels(lf_nums, rotation=90)
axes.set_xlim(ind[0] - 1, ind[-1]+1)

fig.savefig('../../predictd_figs/hyperparams_dist.pdf', bbox_inches='tight')

axes.set_title('Test0, Valid2, Same Genomic Windows\n')
fig.show()


# ###New Hyperparams

# In[10]:

#ri2 4.122
new_ri2_keys = s3_library.glob_keys('predictd', 'PREDICTD/trained_predictd_models/test0/valid_fold?/iter_errs.orig.txt')
new_ri2_errs_high = []
new_ri2_errs_sgd = []
for key in new_ri2_keys:
    with smart_open.smart_open(key) as lines_in:
        iter_errs = numpy.loadtxt(lines_in, skiprows=1, usecols=(1,2,3,4))
    new_ri2_errs_sgd.append(iter_errs[numpy.argsort(iter_errs[:-1,2])[0]])
    new_ri2_errs_high.append(iter_errs[-1])
new_ri2_errs_sgd = numpy.array(new_ri2_errs_sgd)
new_ri2_errs_high = numpy.array(new_ri2_errs_high)


# In[11]:

#ri2 0.4122
new_ri2_keys_med = s3_library.glob_keys('predictd', 'PREDICTD/trained_predictd_models/test0/valid_fold?/iter_errs.txt')
new_ri2_errs_med = []
for key in new_ri2_keys_med:
    with smart_open.smart_open(key) as lines_in:
        iter_errs = numpy.loadtxt(lines_in, skiprows=1, usecols=(1,2,3,4))
    new_ri2_errs_med.append(iter_errs)
new_ri2_errs_med = numpy.array(new_ri2_errs_med)


# In[12]:

#ri2 0.04122
new_ri2_keys_low = s3_library.glob_keys('predictd', 'PREDICTD/hyperparameter_search/test_ri2_0.04122/test0/valid_fold?/iter_errs.txt')
new_ri2_errs_low = []
for key in new_ri2_keys_low:
    with smart_open.smart_open(key) as lines_in:
        iter_errs = numpy.loadtxt(lines_in, skiprows=1, usecols=(1,2,3,4))
    new_ri2_errs_low.append(iter_errs)
new_ri2_errs_low = numpy.array(new_ri2_errs_low)


# In[13]:

ind = numpy.arange(8)
fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(16,8))
width = 0.18
axes.bar(ind, new_ri2_errs_sgd[:,2], width, align='center', color='blue', label='SGD')
axes.bar(ind+width, new_ri2_errs_high[:,2], width, align='center', color='red', label='ri2=4.12')
axes.bar(ind+(2* width), new_ri2_errs_med[:,2], width, align='center', color='k', label='ri2=4.12e-1')
axes.bar(ind+(3* width), new_ri2_errs_low[:,2], width, align='center', color='green', label='ri2=4.12e-2')
axes.set_ylabel('Validation MSE', fontsize=18)
axes.set_xlabel('Validation set', fontsize=18)
axes.set_xticks(ind + (1.5*width))
axes.set_xticklabels(ind)
axes.legend(bbox_to_anchor=[1.12, 1.0])

fig.savefig('../../predictd_figs/validation_mse_by_ri2.pdf', bbox_inches='tight')

axes.set_title('Test Set 0', fontsize=22)
fig.show()


# ##Example of training error curves

# In[14]:

#get iter_errs file
sgd_iter_errs_url = 's3://predictd/PREDICTD/trained_predictd_models/test0/valid_fold0/iter_errs.orig.txt'
with smart_open.smart_open(sgd_iter_errs_url) as iter_errs_in:
    headers = iter_errs_in.readline().strip()
    sgd_iter_errs = numpy.loadtxt(iter_errs_in)[:-1]
order2_iter_errs_url = 's3://predictd/PREDICTD/trained_predictd_models/test0/valid_fold0/iter_errs.txt'
with smart_open.smart_open(order2_iter_errs_url) as iter_errs_in:
    order2_iter_errs = numpy.loadtxt(iter_errs_in, skiprows=1)
iter_errs = numpy.vstack([sgd_iter_errs, order2_iter_errs])


# In[15]:

fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(18,8))

#plot training error
lns1 = axes[0].plot(iter_errs[:-1,0], iter_errs[:-1,2], color='blue', label='Training MSE', linewidth=3.5)
lns1.append(axes[0].axhline(iter_errs[-1,2], color='blue', linestyle=':', linewidth=2, label='Final training MSE'))

#plot validation error
lns2 = axes[0].plot(iter_errs[:-1,0], iter_errs[:-1,3], color='red', label='Validation MSE', linewidth=3.5)
lns2.append(axes[0].axhline(iter_errs[-1,3], color='darkred', linestyle=':', linewidth=2, label='Final validation MSE'))

#show location of minimum validation MSE
min_iter = numpy.where(iter_errs[:-1,3] == numpy.min(iter_errs[:-1,3]))[0][0]
print(iter_errs[min_iter])
print(iter_errs[-1])
scatters = axes[0].scatter([min_iter, min_iter], iter_errs[min_iter, 2:4] + 0.003, color='orange', marker='v', s=100, label='Min SGD iteration')

#combine into one legend (thanks: https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend)
lns = lns1+lns2+[scatters]
labs = [l.get_label() for l in lns]
axes[0].legend(lns, labs, loc='best', fontsize=16)
axes[0].set_xlim((-10,300))

#label MSE plot
axes[0].set_xlabel('Parallel SGD iteration', fontsize=18)
axes[0].set_ylabel('Mean squared error', fontsize=18)

#plot objective
lns3 = axes[1].plot(iter_errs[:-1,0], iter_errs[:-1,1], color='purple', label='Objective', linewidth=3.5)
axes[1].scatter(min_iter, iter_errs[min_iter, 1] + 25000, color='orange', marker='v', s=100, label='Min SGD iteration')
axes[1].legend(loc='best', fontsize=16)
axes[1].set_ylabel('Objective value', fontsize=18)
axes[1].set_xlabel('Parallel SGD iteration', fontsize=18)
axes[1].set_xlim((-10,300))

fig.tight_layout()
fig.savefig('../../predictd_figs/error_curves.pdf', bbox_inches='tight')


# In[15]:



