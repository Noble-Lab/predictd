
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
#%load_ext autoreload
#%autoreload 1

import itertools
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import pyplot
from matplotlib.colors import LogNorm
import numpy
import pickle
import random
from scipy import stats
import scipy.sparse as sps
import smart_open
import sys
import ternary

matplotlib.rcParams['figure.facecolor'] = 'w'

#%aimport ternary

sys.path.append('..')
import s3_library


# ##Define Functions

# In[2]:

def plot_ternary(imp_vals, chrimp_vals, me_vals, ax, measure_name='Measure', ax_linewidth=3.0,
                 ax_label_offset=0.13, ax_label_fsize=20, tick_fontsize=16, ticklabel_offset=0.02, 
                 gridalpha=0.4, plotguides=7, guide_linewidth=4.0, ticklength=0.01, tickwidth=2.0, xlabels=None):
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.0)
    axes_colors={'b':'#0070fd', 'l':'#aa0000', 'r':'#d0af20'}
    tax.boundary(linewidth=ax_linewidth, axes_colors=axes_colors)
    tax.gridlines(color='k', multiple=0.1, alpha=gridalpha)
    if xlabels is None:
        tax.bottom_axis_label('PREDICTD {!s} %'.format(measure_name), fontsize=ax_label_fsize, offset=0.08-ax_label_offset)
        tax.right_axis_label('ChromImpute {!s} %'.format(measure_name), fontsize=ax_label_fsize, offset=ax_label_offset)
        tax.left_axis_label('Main Effects {!s} %'.format(measure_name), fontsize=ax_label_fsize, offset=ax_label_offset)
    else:
        tax.bottom_axis_label(xlabels[0].format(measure_name), fontsize=ax_label_fsize, offset=0.08-ax_label_offset)
        tax.right_axis_label(xlabels[1].format(measure_name), fontsize=ax_label_fsize, offset=ax_label_offset)
        tax.left_axis_label(xlabels[2].format(measure_name), fontsize=ax_label_fsize, offset=ax_label_offset)

    guide_ln = numpy.arange(0,1,0.01)
    guide_ln2 = (1.0 - guide_ln)/2.0
    guides = numpy.hstack([guide_ln[:,None], guide_ln2[:,None], guide_ln2[:,None]])
    if plotguides & 1:
        tax.plot(guides[:,(0,1,2)], linestyle='--', color=axes_colors['b'], linewidth=guide_linewidth, alpha=1, zorder=1)
        tax.plot(guides[:,(0,1,2)], linestyle='-', color=axes_colors['r'], linewidth=guide_linewidth, alpha=1, zorder=0)
    if plotguides & 2:
        tax.plot(guides[:,(1,0,2)], linestyle='--', color=axes_colors['l'], linewidth=guide_linewidth, alpha=1, zorder=1)
        tax.plot(guides[:,(1,0,2)], linestyle='-', color=axes_colors['r'], linewidth=guide_linewidth, alpha=1, zorder=0)
    if plotguides & 4:
        tax.plot(guides[:,(1,2,0)], linestyle='--', color=axes_colors['l'], linewidth=guide_linewidth, alpha=1, zorder=1)
        tax.plot(guides[:,(1,2,0)], linestyle='-', color=axes_colors['b'], linewidth=guide_linewidth, alpha=1, zorder=0)

    #shade areas on plot where each imputation method "wins"
    if measure_name in ['MSEglobal', 'MSE1obs', 'MSE1imp', 'MSE1impchrimp', 'MSE1imppred', 'MSE1impme']:
        #PREDICTD LOW
        xs,ys = ternary.helpers.project_sequence([[0,0,1], [1.0/3, 1.0/3, 1.0/3], [0,1,0]])
        patch = Polygon(numpy.array(zip(xs, ys)), True, color=axes_colors['l'], zorder=-1, alpha=0.25)
        ax.add_patch(patch)
        #MAIN EFFECTS LOW
        xs,ys = ternary.helpers.project_sequence([[1,0,0], [1.0/3, 1.0/3, 1.0/3], [0,1,0]])
        patch = Polygon(numpy.array(zip(xs, ys)), True, color=axes_colors['r'], zorder=-1, alpha=0.25)
        ax.add_patch(patch)
        #CHROM IMPUTE LOW
        xs,ys = ternary.helpers.project_sequence([[1,0,0], [1.0/3, 1.0/3, 1.0/3], [0,0,1]])
        patch = Polygon(numpy.array(zip(xs, ys)), True, color=axes_colors['b'], zorder=-1, alpha=0.25)
        ax.add_patch(patch)
    elif measure_name in ['GWcorrasinh', 'GWspcorr', 'GWcorr', 'Match1', 'Catch1obs', 'Catch1imp', 'AucObs1', 'AucImp1', 'CatchPeakObs']:
        #MAIN EFFECTS HIGH
        xs,ys = ternary.helpers.project_sequence([[0,0,1], [0.5,0,0.5], [1.0/3,1.0/3,1.0/3], [0,0.5,0.5]])
        patch = Polygon(numpy.array(zip(xs, ys)), True, color=axes_colors['r'], zorder=-1, alpha=0.25)
        ax.add_patch(patch)
        #PREDICTD HIGH
        xs,ys = ternary.helpers.project_sequence([[1,0,0], [0.5,0,0.5], [1.0/3,1.0/3,1.0/3], [0.5,0.5,0]])
        patch = Polygon(numpy.array(zip(xs, ys)), True, color=axes_colors['l'], zorder=-1, alpha=0.25)
        ax.add_patch(patch)
        #CHROM IMPUTE HIGH
        xs,ys = ternary.helpers.project_sequence([[0,1,0], [0,0.5,0.5], [1.0/3,1.0/3,1.0/3], [0.5,0.5,0]])
        patch = Polygon(numpy.array(zip(xs, ys)), True, color=axes_colors['b'], zorder=-1, alpha=0.25)
        ax.add_patch(patch)

    points = numpy.array([ternary.helpers.normalize(elt) for elt in zip(imp_vals, chrimp_vals, me_vals)])
    tax.scatter(points, s=5, zorder=2)

    tax.ticks(axis='lbr', multiple=0.1, linewidth=tickwidth, offset=ticklength, fontsize=tick_fontsize, 
              ticklabel_offset=ticklabel_offset)
    tax.clear_matplotlib_ticks()

def plot_corr(imp_vals, chrimp_vals, me_vals, ax, width=0.3, title=None):
    xlocs = numpy.arange(1,4)
    xlabels = ['PREDICTD vs\nChromImpute', 
               'PREDICTD vs\nMain Effects', 
               'Main Effects vs\nChromImpute']

    corr_r, corr_p = list(zip(*[stats.pearsonr(imp_vals, chrimp_vals),
                                stats.pearsonr(imp_vals, me_vals),
                                stats.pearsonr(me_vals, chrimp_vals)]))
    ax.bar(xlocs, corr_r, width=width, align='center')
    for idx, pval in enumerate(corr_p):
        ax.text(xlocs[idx], max(corr_r) + 0.03, '$p=${0:.2e}'.format(pval), 
                ha='center', va='center', size=12)
    ax.set_ylim(0,numpy.max(corr_r) + 0.1)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xlabels, fontsize=14)
    ax.set_ylabel('Pearson\'s r', fontsize=18)

def significance_bar(ax, start, end, height, displaystring, linewidth=1.2,
                     markersize=8, boxpad=0.3, fontsize=15, color='k'):
    '''Thanks - http://stackoverflow.com/questions/33873176/
    how-to-add-significance-levels-on-bar-graph-using-pythons-matplotlib
    '''
    # draw a line with downticks at the ends
    ax.plot([start,end], [height]*2, '-',
             color=color, lw=linewidth, marker=matplotlib.markers.TICKDOWN, 
             markeredgewidth=linewidth, markersize=markersize)
    # draw the text with a bounding box covering up the line
    ax.text(0.5*(start+end), height, displaystring, 
             ha='center', va='center', 
             bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),
             size=fontsize)

def plot_mse(imp_vals, chrimp_vals, me_vals, ax, width=0.3, title=None):
    xlocs = numpy.arange(1,4)
    xlabels = ['PREDICTD vs\nChromImpute', 
               'PREDICTD vs\nMain Effects', 
               'ChromImpute vs\nMain Effects']

#    mse_vals = [numpy.mean((imp_vals - chrimp_vals) ** 2),
#                numpy.mean((imp_vals - me_vals) ** 2),
#                numpy.mean((chrimp_vals - me_vals) ** 2)]
    mse_vals = [numpy.log(((imp_vals - chrimp_vals) ** 2) + 1e-10),
                numpy.log(((imp_vals - me_vals) ** 2) + 1e-10),
                numpy.log(((chrimp_vals - me_vals) ** 2) + 1e-10)]
#    ax.bar(xlocs, mse_vals, align='center')
#    ax.boxplot(mse_vals, showmeans=True)
    ax.violinplot(mse_vals, showmeans=True)

    pvals = [stats.wilcoxon(mse_vals[0], mse_vals[1])[1],
             stats.wilcoxon(mse_vals[0], mse_vals[2])[1],
             stats.wilcoxon(mse_vals[1], mse_vals[2])[1]]
    significance_bar(ax, xlocs[0], xlocs[1], numpy.max(mse_vals) + 1,
                     'p={0:.2e}'.format(pvals[0]), fontsize=12)
    significance_bar(ax, xlocs[0], xlocs[2], numpy.max(mse_vals) + 2.5,
                     'p={0:.2e}'.format(pvals[1]), fontsize=12)
    significance_bar(ax, xlocs[1], xlocs[2], numpy.max(mse_vals) + 4,
                     'p={0:.2e}'.format(pvals[2]), fontsize=12)

    ax.set_ylim(numpy.min(mse_vals) - 1, numpy.max(mse_vals) + 5)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xlabels, fontsize=14)
    ax.set_ylabel('ln(Squared Error)', fontsize=18)

def plot_fold_change(imp_vals, chrimp_vals, me_vals, ax, width=0.3, title=None):
    xlocs = numpy.arange(1,4)
    xlabels = ['PREDICTD vs\nChromImpute', 
               'PREDICTD vs\nMain Effects', 
               'Main Effects vs\nChromImpute']

#    mse_vals = [numpy.mean((imp_vals - chrimp_vals) ** 2),
#                numpy.mean((imp_vals - me_vals) ** 2),
#                numpy.mean((chrimp_vals - me_vals) ** 2)]
    mse_vals = [numpy.log(imp_vals + 1e-10)-numpy.log(chrimp_vals + 1e-10),
                numpy.log(imp_vals + 1e-10)-numpy.log(me_vals + 1e-10),
                numpy.log(me_vals + 1e-10)-numpy.log(chrimp_vals + 1e-10)]
#    ax.bar(xlocs, mse_vals, align='center')
#    ax.boxplot(mse_vals, showmeans=True)
    ax.violinplot(mse_vals, showmeans=True)

#    pvals = [stats.wilcoxon(mse_vals[0], numpy.random.normal(scale=numpy.std(mse_vals[0]), size=mse_vals[0].shape)),
#             stats.wilcoxon(mse_vals[1], numpy.random.normal(scale=numpy.std(mse_vals[1]), size=mse_vals[1].shape)),
#             stats.wilcoxon(mse_vals[2], numpy.random.normal(scale=numpy.std(mse_vals[2]), size=mse_vals[2].shape))]
    pvals = [(stats.ttest_1samp(mse_vals[0], 0), numpy.std(mse_vals[0])),
             (stats.ttest_1samp(mse_vals[1], 0), numpy.std(mse_vals[1])),
             (stats.ttest_1samp(mse_vals[2], 0), numpy.std(mse_vals[2]))]
#    significance_bar(ax, xlocs[0], xlocs[1], numpy.max(mse_vals) + 0.2,
#                     'p={0:.2e}'.format(pvals[0]), fontsize=12)
#    significance_bar(ax, xlocs[0], xlocs[2], numpy.max(mse_vals) + 0.6,
#                     'p={0:.2e}'.format(pvals[1]), fontsize=12)
#    significance_bar(ax, xlocs[1], xlocs[2], numpy.max(mse_vals) + 1,
#                     'p={0:.2e}'.format(pvals[2]), fontsize=12)
    for idx, (pval, stdev) in enumerate(pvals):
        ax.text(xlocs[idx], numpy.max(mse_vals) + 0.06, '$\sigma=${0:.2e},\n$t=${1:.2e},\n$p=${2:.2e}'.format(stdev, pval[0], pval[1]), 
                ha='center', va='bottom', size=12)
    ax.set_ylim(numpy.min(mse_vals) - 0.5, numpy.max(mse_vals) + 1.1)
    ax.axhline(0, linestyle='--')
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xlabels, fontsize=14)
    ax.set_ylabel('Log ratio', fontsize=18)


# ##Blank Ternary Plot

# In[3]:

fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(10,10))
imp_vals = [1]
chrimp_vals = [1]
me_vals = [1]
plot_ternary(imp_vals, chrimp_vals, me_vals, axes, measure_name='MSEglobal', gridalpha=1.0,
             ticklength=0.03, tickwidth=2.0, ticklabel_offset=0.03, ax_label_offset=0.15, ax_label_fsize=24, tick_fontsize=20,
             ax_linewidth=8.0, guide_linewidth=6.0, plotguides=0)
fig.savefig('../../predictd_figs/blank_ternary.pdf', bbox_inches='tight')


# ##Load In Quality Measures

# In[4]:

#standard test set data
#field order: expt_id, expt_coord, mse1altimp, mse1altimprecip
with smart_open.smart_open('s3://predictd/MainEffects/quality_measures.2D_me_all.predictd_alt.txt', 'r') as lines_in:
    me2d_alt_pred = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/ChromImpute/quality_measures.chromimpute_all.2D_me_alt.txt', 'r') as lines_in:
    chrimp_alt_me2d = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.predictd_all.chromimpute_alt.txt', 'r') as lines_in:
    pred_alt_chrimp = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.avg_predictd_chromimpute.predictd_alt.txt', 'r') as lines_in:
    pred_chrimp_avg = numpy.array([elt.strip().split() for elt in lines_in])


# In[5]:

#low ct data
with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.Brain_Anterior_Caudate_H3K4me3.chromimpute_alt.txt', 'r') as lines_in:
    brain_anterior_caudate_h3k4me3 = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.CD3_Primary_Cells_Cord_BI_H3K4me1.chromimpute_alt.txt', 'r') as lines_in:
    cd3_cord_h3k4me1 = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.CD3_Primary_Cells_Cord_BI_H3K4me3.chromimpute_alt.txt', 'r') as lines_in:
    cd3_cord_h3k4me3 = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.CD3_Primary_Cells_Cord_BI_H3K4me3_H3K9me3.chromimpute_alt.txt', 'r') as lines_in:
    cd3_cord_h3k4me3_h3k9me3 = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.Fetal_Muscle_Trunk_H3K4me3.chromimpute_alt.txt', 'r') as lines_in:
    fetal_muscle_h3k4me3 = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.GM12878_Lymphoblastoid_H3K4me3.chromimpute_alt.txt', 'r') as lines_in:
    gm12878_h3k4me3 = numpy.array([elt.strip().split() for elt in lines_in])

with smart_open.smart_open('s3://predictd/PREDICTD/quality_measures.Lung_H3K4me3.chromimpute_alt.txt', 'r') as lines_in:
    lung_h3k4me3 = numpy.array([elt.strip().split() for elt in lines_in])


# ##Plot quality measures with ternary plots

# In[6]:

#test sort order of the different quality measures files (they should be the same)
print('PREDICTD same order as ChromImpute: {!s}'.format(numpy.array_equal(pred_alt_chrimp[:,0], chrimp_alt_me2d[:,0])))
print('PREDICTD same order as ME2D: {!s}'.format(numpy.array_equal(pred_alt_chrimp[:,0], me2d_alt_pred[:,0])))


# In[7]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

#for some reason setting frameon=False causes plotting to ignore the matplotlib.rcParams['figure.facecolor'] = 'w' setting, which
#is required to avoid transparency issues when converting the image to pdf for the paper
#fig, axes = pyplot.subplots(frameon=False, nrows=13, ncols=3, figsize=(24,104), gridspec_kw={'width_ratios':[1.5,1,1]})
fig, axes = pyplot.subplots(nrows=13, ncols=3, figsize=(24,104), gridspec_kw={'width_ratios':[1.5,1,1]})

for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5), 
                                           ('MSE1impchrimp', 18), ('MSE1imppred', 18), ('MSE1impme', 18),
                                           ('GWcorrasinh', 6), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    if label == 'MSE1impchrimp':
        imp_vals = pred_alt_chrimp[:,18].flatten().astype(float)
        me_vals = chrimp_alt_me2d[:,19].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[:,5].flatten().astype(float)
    elif label == 'MSE1imppred':
        imp_vals = pred_alt_chrimp[:,5].flatten().astype(float)
        me_vals = me2d_alt_pred[:,18].flatten().astype(float)
        chrimp_vals = pred_alt_chrimp[:,19].flatten().astype(float)
    elif label == 'MSE1impme':
        imp_vals = me2d_alt_pred[:,19].flatten().astype(float)
        me_vals = me2d_alt_pred[:,5].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[:,18].flatten().astype(float)
    else:
        imp_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
        me_vals = me2d_alt_pred[:,col_idx].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)

#    plot_ternary(imp_vals, chrimp_vals, me_vals, axes[ax_idx,0], measure_name=label)
    plot_ternary(imp_vals, chrimp_vals, me_vals, axes[ax_idx,0], measure_name=label, gridalpha=1.0,
                 ticklength=0.03, tickwidth=2.0, ticklabel_offset=0.03, ax_label_offset=0.15, ax_label_fsize=24, tick_fontsize=20,
                 ax_linewidth=8.0, guide_linewidth=6.0, plotguides=0)
    plot_corr(imp_vals, chrimp_vals, me_vals, axes[ax_idx,1], width=0.5)
    plot_fold_change(imp_vals, chrimp_vals, me_vals, axes[ax_idx,2])

#fig.patch.set_visible(False)
pyplot.show()
fig.savefig('../../predictd_figs/all_quality_measures.pdf', bbox_inches='tight')
pyplot.close(fig)
del(fig)


# ###Same figure, but with the final test experiments (test fold 0 experiments not used for hyperparameter tuning)

# In[8]:

not_final_test = s3_library.get_pickle_s3('final-rev', 'trained_models/not_final_test_subset.pickle')
not_final_test_flat = ((numpy.array(not_final_test[0]) * 24) + numpy.array(not_final_test[1]))
test0_coords = numpy.where(~s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/folds.5.8.pickle')[0]['test'])
test0_flat = (test0_coords[0] * 24) + test0_coords[1]
to_use_flat = [elt for elt in test0_flat if elt not in not_final_test_flat]
print(len(to_use_flat))


# In[9]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

#fig, axes = pyplot.subplots(frameon=False, nrows=3, ncols=3, figsize=(24,24), gridspec_kw={'width_ratios':[1,1,1]})

pred_coords_flat = (pred_alt_chrimp[:,1].astype(int) * 24) + pred_alt_chrimp[:,2].astype(int)
final_test_coords = [numpy.where(pred_coords_flat == elt)[0][0] for elt in to_use_flat]

#for some reason setting frameon=False causes plotting to ignore the matplotlib.rcParams['figure.facecolor'] = 'w' setting, which
#is required to avoid transparency issues when converting the image to pdf for the paper
#fig, axes = pyplot.subplots(frameon=False, nrows=13, ncols=3, figsize=(24,104), gridspec_kw={'width_ratios':[1.5,1,1]})
fig, axes = pyplot.subplots(nrows=13, ncols=3, figsize=(24,104), gridspec_kw={'width_ratios':[1.5,1,1]})

for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5), 
                                           ('MSE1impchrimp', 18), ('MSE1imppred', 18), ('MSE1impme', 18),
                                           ('GWcorrasinh', 6), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    if label == 'MSE1impchrimp':
        imp_vals = pred_alt_chrimp[final_test_coords,18].flatten().astype(float)
        me_vals = chrimp_alt_me2d[final_test_coords,19].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[final_test_coords,5].flatten().astype(float)
    elif label == 'MSE1imppred':
        imp_vals = pred_alt_chrimp[final_test_coords,5].flatten().astype(float)
        me_vals = me2d_alt_pred[final_test_coords,18].flatten().astype(float)
        chrimp_vals = pred_alt_chrimp[final_test_coords,19].flatten().astype(float)
    elif label == 'MSE1impme':
        imp_vals = me2d_alt_pred[final_test_coords,19].flatten().astype(float)
        me_vals = me2d_alt_pred[final_test_coords,5].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[final_test_coords,18].flatten().astype(float)
    else:
        imp_vals = pred_alt_chrimp[final_test_coords,col_idx].flatten().astype(float)
        me_vals = me2d_alt_pred[final_test_coords,col_idx].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[final_test_coords,col_idx].flatten().astype(float)

#    plot_ternary(imp_vals, chrimp_vals, me_vals, axes[ax_idx,0], measure_name=label)
    plot_ternary(imp_vals, chrimp_vals, me_vals, axes[ax_idx,0], measure_name=label, gridalpha=1.0,
                 ticklength=0.03, tickwidth=2.0, ticklabel_offset=0.03, ax_label_offset=0.15, ax_label_fsize=24, tick_fontsize=20,
                 ax_linewidth=8.0, guide_linewidth=6.0, plotguides=0)
    plot_corr(imp_vals, chrimp_vals, me_vals, axes[ax_idx,1], width=0.5)
    plot_fold_change(imp_vals, chrimp_vals, me_vals, axes[ax_idx,2])

#fig.patch.set_visible(False)
pyplot.show()
fig.savefig('../../predictd_figs/all_quality_measures_final_test.pdf', bbox_inches='tight')
pyplot.close(fig)
del(fig)


# In[10]:

#print measure comparisons to file
data_rows = []
for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5), 
                                           ('MSE1impchrimp', 18), ('MSE1imppred', 18), ('MSE1impme', 18),
                                           ('GWcorrasinh', 6), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    if label == 'MSE1impchrimp':
        imp_vals = pred_alt_chrimp[:,18].flatten().astype(float)
        me_vals = chrimp_alt_me2d[:,19].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[:,5].flatten().astype(float)
    elif label == 'MSE1imppred':
        imp_vals = pred_alt_chrimp[:,5].flatten().astype(float)
        me_vals = me2d_alt_pred[:,18].flatten().astype(float)
        chrimp_vals = pred_alt_chrimp[:,19].flatten().astype(float)
    elif label == 'MSE1impme':
        imp_vals = me2d_alt_pred[:,19].flatten().astype(float)
        me_vals = me2d_alt_pred[:,5].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[:,18].flatten().astype(float)
    else:
        imp_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
        me_vals = me2d_alt_pred[:,col_idx].flatten().astype(float)
        chrimp_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)

    corr_r, corr_p = list(zip(*[stats.pearsonr(imp_vals, chrimp_vals),
                                stats.pearsonr(imp_vals, me_vals),
                                stats.pearsonr(chrimp_vals, me_vals)]))

    fold_change = [numpy.log(imp_vals + 1e-10)-numpy.log(chrimp_vals + 1e-10),
                   numpy.log(imp_vals + 1e-10)-numpy.log(me_vals + 1e-10),
                   numpy.log(me_vals + 1e-10)-numpy.log(chrimp_vals + 1e-10)]

    pvals = [(stats.ttest_1samp(fold_change[0], 0), numpy.std(fold_change[0])),
             (stats.ttest_1samp(fold_change[1], 0), numpy.std(fold_change[1])),
             (stats.ttest_1samp(fold_change[2], 0), numpy.std(fold_change[2]))]
    data_rows.append([label, corr_r[0], numpy.mean(fold_change[0]), pvals[0][1], pvals[0][0][0], pvals[0][0][1],
                      corr_r[1], numpy.mean(fold_change[1]), pvals[1][1], pvals[1][0][0], pvals[1][0][1],
                      corr_r[2], numpy.mean(fold_change[2]), pvals[2][1], pvals[2][0][0], pvals[2][0][1]])
fname = './predictd_vs_chromimpute_compare_stats.tab'
headers = ['quality measure', 'pred vs chrimp correlation', 'pred vs chrimp mean log ratio', 'pred vs chrimp std log ratio',
           'pred vs chrimp log ratio one-sample t', 'pred vs chrimp log ratio one-sample t p-val', 'pred vs me correlation', 
           'pred vs me mean log ratio', 'pred vs me std log ratio', 'pred vs me log ratio one-sample t', 
           'pred vs me log ratio one-sample t p-val', 'me vs chrimp correlation', 'me vs chrimp mean log ratio', 
           'me vs chrimp std log ratio', 'me vs chrimp log ratio one-sample t', 
           'me vs chrimp log ratio one-sample t p-val']
numpy.savetxt(fname, numpy.array(data_rows, dtype=object), fmt='%s', header=', '.join(headers), delimiter='\t')


# In[11]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})
#matplotlib.rcParams['figure.facecolor'] = 'w'


metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

#for some reason setting frameon=False causes plotting to ignore the matplotlib.rcParams['figure.facecolor'] = 'w' setting, which
#is required to avoid transparency issues when converting the image to pdf for the paper
#fig, axes = pyplot.subplots(frameon=False, nrows=1, ncols=5, figsize=(50,8), facecolor='w')
fig, axes = pyplot.subplots(nrows=1, ncols=5, figsize=(50,8))

for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), #('MSE1obs', 4), ('MSE1imp', 5),
                                           ('GWcorrasinh', 6), #('GWspcorr', 8), ('GWcorr', 10), ('Match1', 12), 
                                           ('Catch1obs', 13), ('Catch1imp', 14), 
                                           #('AucObs1', 15), ('AucImp1', 16), 
                                           ('CatchPeakObs', 17)]):
    imp_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
    me_vals = me2d_alt_pred[:,col_idx].flatten().astype(float)
    chrimp_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)

    plot_ternary(imp_vals, chrimp_vals, me_vals, axes[ax_idx], measure_name=label.replace('asinh', ''), gridalpha=1.0,
                 ticklength=0.03, tickwidth=2.0, ticklabel_offset=0.03, ax_label_offset=0.15, ax_label_fsize=24, tick_fontsize=20,
                 ax_linewidth=8.0, guide_linewidth=6.0, plotguides=0)

#fig.patch.set_visible(False)
pyplot.show()
fig.savefig('../../predictd_figs/quality_measures_main_fig.pdf', bbox_inches='tight')
pyplot.close(fig)
del(fig)


# In[12]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

#for some reason setting frameon=False causes plotting to ignore the matplotlib.rcParams['figure.facecolor'] = 'w' setting, which
#is required to avoid transparency issues when converting the image to pdf for the paper
#fig, axes = pyplot.subplots(frameon=False, nrows=1, ncols=5, figsize=(50,8))
fig, axes = pyplot.subplots(nrows=1, ncols=5, figsize=(50,8))


for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), #('MSE1obs', 4), ('MSE1imp', 5),
                                           ('GWcorrasinh', 6), #('GWspcorr', 8), ('GWcorr', 10), ('Match1', 12), 
                                           ('Catch1obs', 13), ('Catch1imp', 14), 
                                           #('AucObs1', 15), ('AucImp1', 16), 
                                           ('CatchPeakObs', 17)]):
#    imp_vals = chrimp_plus_imputed_metrics[imp_sort,col_idx].flatten().astype(float)
    imp_vals = pred_chrimp_avg[:,col_idx].flatten().astype(float)
    me_vals = me2d_alt_pred[:,col_idx].flatten().astype(float)
    chrimp_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)

    plot_ternary(imp_vals, chrimp_vals, me_vals, axes[ax_idx], measure_name=label.replace('asinh', ''), gridalpha=1.0,
                 ticklength=0.03, tickwidth=2.0, ticklabel_offset=0.03, ax_label_offset=0.15, ax_label_fsize=24, tick_fontsize=20,
                 ax_linewidth=8.0, guide_linewidth=6.0, plotguides=0,
                 xlabels=['     Avg(PREDICTD, ChromImpute) {!s} %', 'ChromImpute {!s} %', 'Main Effects {!s} %'])
#    plot_ternary(imp_vals, chrimp_vals, me_vals, axes[ax_idx], measure_name=label.replace('asinh', ''),
#                 xlabels=['(PREDICTD + ChromImpute)/2 {!s} %', 'ChromImpute {!s} %', 'Main Effects {!s} %'])

#fig.patch.set_visible(False)
pyplot.show()
fig.savefig('../../predictd_figs/quality_measures_main_fig_avg.pdf', bbox_inches='tight')
pyplot.close(fig)
del(fig)


# The performance metrics were collected based on the trained models at:
# 
# s3://final-rev/trained_models_low_ri2_0.412178570144

# ##Performance metrics by assay and comparison to main effects baseline

# In[13]:

sorted_mse = numpy.argsort(pred_alt_chrimp[:,3].flatten().astype(float))
print('Avg PREDICTD MSE: {:.4f}'.format(numpy.mean(pred_alt_chrimp[sorted_mse,3].flatten().astype(float))))
print('Min Data Set: {!s}'.format(pred_alt_chrimp[sorted_mse[0],0]))
print('Min PREDICTD MSE: ' + ', '.join(['{:.4f}'.format(elt) for elt in pred_alt_chrimp[sorted_mse[:10],3].flatten().astype(float)]))
print('Max Data Set: {!s}'.format(pred_alt_chrimp[sorted_mse[-1],0]))
print('Max PREDICTD MSE: ' + ', '.join(['{:.4f}'.format(elt) for elt in pred_alt_chrimp[sorted_mse[-10:],3].flatten().astype(float)]))


# In[14]:

sorted_mse = numpy.argsort(chrimp_alt_me2d[:,3].flatten().astype(float))
print('Avg ChromImpute MSE: {:.4f}'.format(numpy.mean(chrimp_alt_me2d[sorted_mse,3].flatten().astype(float))))
print('Min Data Set: {!s}'.format(pred_alt_chrimp[sorted_mse[0],0]))
print('Min ChromImpute MSE: ' + ', '.join(['{:.4f}'.format(elt) for elt in chrimp_alt_me2d[sorted_mse[:10],3].flatten().astype(float)]))
print('Max Data Set: {!s}'.format(pred_alt_chrimp[sorted_mse[-1],0]))
print('Max ChromImpute MSE: ' + ', '.join(['{:.4f}'.format(elt) for elt in chrimp_alt_me2d[sorted_mse[-10:],3].flatten().astype(float)]))


# In[15]:

col_idx = 6
sorted_mse = numpy.argsort(pred_alt_chrimp[:,col_idx].flatten().astype(float))
print('Avg PREDICTD GWcorr: {:.4f}'.format(numpy.mean(pred_alt_chrimp[sorted_mse,col_idx].flatten().astype(float))))
print('Min Data Set: {!s}'.format(pred_alt_chrimp[sorted_mse[0],0]))
print('Min PREDICTD GWcorr: ' + ', '.join(['{:.4f}'.format(elt) for elt in pred_alt_chrimp[sorted_mse[:10],col_idx].flatten().astype(float)]))
print('Max Data Set: {!s}'.format(pred_alt_chrimp[sorted_mse[-1],0]))
print('Max PREDICTD GWcorr: ' + ', '.join(['{:.4f}'.format(elt) for elt in pred_alt_chrimp[sorted_mse[-10:],col_idx].flatten().astype(float)]))


# In[16]:

col_idx = 17
sorted_mse = numpy.argsort(pred_alt_chrimp[:,col_idx].flatten().astype(float))
print('Avg PREDICTD CatchPeakObs: {:.4f}'.format(numpy.mean(pred_alt_chrimp[sorted_mse,col_idx].flatten().astype(float))))
print('Min Data Set: {!s}'.format(pred_alt_chrimp[sorted_mse[0],0]))
print('Min PREDICTD CatchPeakObs: ' + ', '.join(['{:.4f}'.format(elt) for elt in pred_alt_chrimp[sorted_mse[:10],col_idx].flatten().astype(float)]))
print('Max Data Set: {!s}'.format(pred_alt_chrimp[sorted_mse[-1],0]))
print('Max PREDICTD CatchPeakObs: ' + ', '.join(['{:.4f}'.format(elt) for elt in pred_alt_chrimp[sorted_mse[-10:],col_idx].flatten().astype(float)]))


# In[17]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

sorted_assays = sorted(set([(elt[0].split('-')[-1], int(elt[1])) for elt in pred_alt_chrimp[:,(0,2)]]), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

gmse_sort = None
for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5), ('MSE1imppred', 18), ('MSE1impme', 18),
                                           ('GWcorrasinh', 6), #('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    if label == 'MSE1imppred':
        imputed_all_vals = pred_alt_chrimp[:,5].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,18].flatten().astype(float)
#        imputed_chrimp_vals = pred_alt_chrimp[:,19].flatten().astype(float)
    elif label == 'MSE1impme':
        imputed_all_vals = me2d_alt_pred[:,19].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,5].flatten().astype(float)
#        imputed_chrimp_vals = chrimp_alt_me2d[:,18].flatten().astype(float)
    else:
        imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,col_idx].flatten().astype(float)
#        imputed_chrimp_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_impall, impall_yerrs, to_plot_impme, impme_yerrs, to_plot_chrimp, chrimp_yerrs = [], [], [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(pred_alt_chrimp[:,2] == str(assay_idx))
        to_plot_impall.append(numpy.mean(imputed_all_vals[coord]))
        impall_yerrs.append(stats.sem(imputed_all_vals[coord]))

        to_plot_impme.append(numpy.mean(imputed_me_vals[coord]))
        impme_yerrs.append(stats.sem(imputed_me_vals[coord]))

#        to_plot_chrimp.append(numpy.mean(imputed_chrimp_vals[coord]))
#        chrimp_yerrs.append(stats.sem(imputed_chrimp_vals[coord]))
    if label == 'MSEglobal':
        gmse_sort = numpy.argsort(to_plot_impall)
    ind = numpy.arange(len(sorted_assays))
#    width = 0.18
    width = 0.25
    edgecol = 'k'
    edgewidth = 0.5
    axes[ax_coord].bar(ind, numpy.array(to_plot_impme)[gmse_sort], width, color='violet', edgecolor=edgecol, linewidth=edgewidth, 
                       label='Main Effects', yerr=numpy.array(impme_yerrs)[gmse_sort], error_kw=dict(ecolor='k'))
    axes[ax_coord].bar(ind + width, numpy.array(to_plot_impall)[gmse_sort], width, color='blue', edgecolor=edgecol, linewidth=edgewidth, 
                       label='PREDICTD', yerr=numpy.array(impall_yerrs)[gmse_sort], error_kw=dict(ecolor='k'))
#    axes[ax_coord].bar(ind + (2 * width), numpy.array(to_plot_chrimp)[gmse_sort], width, color='red', edgecolor=edgecol, linewidth=edgewidth, 
#                       label='ChromImpute', yerr=numpy.array(chrimp_yerrs)[gmse_sort], error_kw=dict(ecolor='k'))

#    axes[ax_coord].set_xticks(ind + (1.5 * width))
    axes[ax_coord].set_xticks(ind + (2 * width))
    axes[ax_coord].set_xticklabels([sorted_assays[idx][0].replace('H2AZ', 'H2A.Z') for idx in gmse_sort], rotation=60, ha='right')
    
    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label.replace('asinh', ''), metric_desc[label][1].replace('with asinh data', '')), fontsize=20)
    if ax_idx == axes.shape[1] - 1:
        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Imputed Assay', fontsize=20)
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
#    axes[ax_coord].set_aspect('equal', 'datalim')

fig.tight_layout()
fig.savefig('../../predictd_figs/quality_measures_barchart_predictd_me.pdf', bbox_inches='tight')


# In[18]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

sorted_assays = sorted(set([(elt[0].split('-')[-1], int(elt[1])) for elt in pred_alt_chrimp[:,(0,2)]]), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

gmse_sort = None
for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5), ('MSE1imppred', 18), ('MSE1impme', 18),
                                           ('GWcorrasinh', 6), #('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    if label == 'MSE1imppred':
        imputed_all_vals = pred_alt_chrimp[:,5].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,18].flatten().astype(float)
        imputed_chrimp_vals = pred_alt_chrimp[:,19].flatten().astype(float)
    elif label == 'MSE1impme':
        imputed_all_vals = me2d_alt_pred[:,19].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,5].flatten().astype(float)
        imputed_chrimp_vals = chrimp_alt_me2d[:,18].flatten().astype(float)
    else:
        imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,col_idx].flatten().astype(float)
        imputed_chrimp_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_impall, impall_yerrs, to_plot_impme, impme_yerrs, to_plot_chrimp, chrimp_yerrs = [], [], [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(pred_alt_chrimp[:,2] == str(assay_idx))
        to_plot_impall.append(numpy.mean(imputed_all_vals[coord]))
        impall_yerrs.append(stats.sem(imputed_all_vals[coord]))

        to_plot_impme.append(numpy.mean(imputed_me_vals[coord]))
        impme_yerrs.append(stats.sem(imputed_me_vals[coord]))

        to_plot_chrimp.append(numpy.mean(imputed_chrimp_vals[coord]))
        chrimp_yerrs.append(stats.sem(imputed_chrimp_vals[coord]))
    if label == 'MSEglobal':
        gmse_sort = numpy.argsort(to_plot_impall)
    ind = numpy.arange(len(sorted_assays))
#    width = 0.18
    width = 0.25
    edgecol = 'k'
    edgewidth = 0.5
    axes[ax_coord].bar(ind, numpy.array(to_plot_impme)[gmse_sort], width, color='violet', edgecolor=edgecol, linewidth=edgewidth, 
                       label='Main Effects', yerr=numpy.array(impme_yerrs)[gmse_sort], error_kw=dict(ecolor='k'))
    axes[ax_coord].bar(ind + width, numpy.array(to_plot_impall)[gmse_sort], width, color='blue', edgecolor=edgecol, linewidth=edgewidth, 
                       label='PREDICTD', yerr=numpy.array(impall_yerrs)[gmse_sort], error_kw=dict(ecolor='k'))
    axes[ax_coord].bar(ind + (2 * width), numpy.array(to_plot_chrimp)[gmse_sort], width, color='red', edgecolor=edgecol, linewidth=edgewidth, 
                       label='ChromImpute', yerr=numpy.array(chrimp_yerrs)[gmse_sort], error_kw=dict(ecolor='k'))

#    axes[ax_coord].set_xticks(ind + (1.5 * width))
    axes[ax_coord].set_xticks(ind + (3 * width))
    axes[ax_coord].set_xticklabels([sorted_assays[idx][0].replace('H2AZ', 'H2A.Z') for idx in gmse_sort], rotation=60, ha='right')
    
    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label.replace('asinh', ''), metric_desc[label][1].replace('with asinh data', '')), fontsize=20)
    if ax_idx == axes.shape[1] - 1:
        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Imputed Assay', fontsize=20)
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
#    axes[ax_coord].set_aspect('equal', 'datalim')

fig.tight_layout()
fig.savefig('../../predictd_figs/quality_measures_barchart_pred_me_chrimp.pdf', bbox_inches='tight')


# ###Plot as distributions instead of bars

# In[59]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

sorted_assays = sorted(set([(elt[0].split('-')[-1], int(elt[1])) for elt in pred_alt_chrimp[:,(0,2)]]), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

gmse_sort = None
for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5), ('MSE1imppred', 18), ('MSE1impme', 18),
                                           ('GWcorrasinh', 6), #('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    if label == 'MSE1imppred':
        imputed_all_vals = pred_alt_chrimp[:,5].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,18].flatten().astype(float)
        imputed_chrimp_vals = pred_alt_chrimp[:,19].flatten().astype(float)
    elif label == 'MSE1impme':
        imputed_all_vals = me2d_alt_pred[:,19].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,5].flatten().astype(float)
        imputed_chrimp_vals = chrimp_alt_me2d[:,18].flatten().astype(float)
    else:
        imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,col_idx].flatten().astype(float)
        imputed_chrimp_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_impall, impall_yerrs, to_plot_impme, impme_yerrs, to_plot_chrimp, chrimp_yerrs = [], [], [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(pred_alt_chrimp[:,2] == str(assay_idx))
        to_plot_impall.append(imputed_all_vals[coord])
        to_plot_impme.append(imputed_me_vals[coord])
        to_plot_chrimp.append(imputed_chrimp_vals[coord])

    to_plot_impall = numpy.array(to_plot_impall, dtype=object)
    to_plot_impme = numpy.array(to_plot_impme, dtype=object)
    to_plot_chrimp = numpy.array(to_plot_chrimp, dtype=object)

    if label == 'MSEglobal':
        gmse_sort = numpy.argsort([numpy.mean(elt) for elt in to_plot_impall])
    ind = numpy.arange(0, len(sorted_assays)*3)
#    width = 0.18
    width = 0.25
    edgecol = 'k'
    edgewidth = 0.5

#    #main effects violins
#    vparts = axes[ax_coord].violinplot(to_plot_impme[gmse_sort], positions=ind[::4], showmeans=True, showextrema=True)
#    for pc in vparts['bodies']:
#        pc.set_color('violet')
##        pc.set_facecolor('violet')
##        pc.set_edgecolor('black')
#    #predictd violins
#    vparts = axes[ax_coord].violinplot(to_plot_impall[gmse_sort], positions=ind[1::4], showmeans=True, showextrema=True)
#    for pc in vparts['bodies']:
#        pc.set_color('blue')
#    #chromimpute violins
#    vparts = axes[ax_coord].violinplot(to_plot_chrimp[gmse_sort], positions=ind[2::4], showmeans=True, showextrema=True)
#    for pc in vparts['bodies']:
#        pc.set_color('red')

    c = 'violet'
    bparts = axes[ax_coord].boxplot(to_plot_impme[gmse_sort], positions=ind[::3], 
                                    boxprops=dict(facecolor=c, color='k'),
                                    capprops=dict(color='k'),
                                    whiskerprops=dict(color='k'),
                                    flierprops=dict(color='k', markerfacecolor='k', markeredgecolor='k'),
                                    medianprops=dict(color='gold', linewidth=2, solid_capstyle="butt"),
#                                    medianprops=dict(color='gold'),
                                    sym='.', patch_artist=True, widths=0.8)
    c = 'blue'
    bparts = axes[ax_coord].boxplot(to_plot_impall[gmse_sort], positions=ind[1::3], 
                                    boxprops=dict(facecolor=c, color='k'),
                                    capprops=dict(color='k'),
                                    whiskerprops=dict(color='k'),
                                    flierprops=dict(color='k', markerfacecolor='k', markeredgecolor='k'),
                                    medianprops=dict(color='gold', linewidth=2, solid_capstyle="butt"),
#                                    medianprops=dict(color='gold'),
                                    sym='.', patch_artist=True, widths=0.8)
#    c = 'red'
#    bparts = axes[ax_coord].boxplot(to_plot_chrimp[gmse_sort], positions=ind[2::4], 
#                                    boxprops=dict(facecolor=c, color='k'),
#                                    capprops=dict(color='k'),
#                                    whiskerprops=dict(color='k'),
#                                    flierprops=dict(color='k', markerfacecolor='k', markeredgecolor='k'),
#                                    medianprops=dict(color='gold', linewidth=2, solid_capstyle="butt"),
##                                    medianprops=dict(color='gold'),
#                                    sym='.', patch_artist=True, widths=0.8)

#    axes[ax_coord].set_xticks(ind + (1.5 * width))
    axes[ax_coord].set_xticks(ind[1::3] - 0.5)
    axes[ax_coord].set_xticklabels([sorted_assays[idx][0].replace('H2AZ', 'H2A.Z') for idx in gmse_sort], rotation=60, ha='right')

    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label.replace('asinh', ''), metric_desc[label][1].replace('with asinh data', '')), fontsize=20)
#    if ax_idx == axes.shape[1] - 1:
#        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
#        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Imputed Assay', fontsize=20)
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
    axes[ax_coord].set_xlim(-1,len(ind))
    
    
    ymin, ymax = (numpy.min([numpy.min(elt) for elt in itertools.chain(*[to_plot_impme, to_plot_impall, to_plot_chrimp])]),
                  numpy.max([numpy.max(elt) for elt in itertools.chain(*[to_plot_impme, to_plot_impall, to_plot_chrimp])]))
    if 'MSE' in label:
        axes[ax_coord].set_yscale('log')
        axes[ax_coord].set_ylim(ymin, ymax)
    if label == 'MSEglobal':
        axes[ax_coord].set_yticks([ymin, 0.1, ymax])
        axes[ax_coord].set_yticklabels(['{:.2e}'.format(ymin), '{:.2e}'.format(0.1), '{:.2e}'.format(ymax)])
#    axes[ax_coord].set_aspect('equal', 'datalim')

fig.tight_layout()
fig.savefig('../../predictd_figs/quality_measures_boxplots_pred_me.pdf', bbox_inches='tight')


# In[57]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(32,32))

sorted_assays = sorted(set([(elt[0].split('-')[-1], int(elt[1])) for elt in pred_alt_chrimp[:,(0,2)]]), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

gmse_sort = None
for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5), ('MSE1imppred', 18), ('MSE1impme', 18),
                                           ('GWcorrasinh', 6), #('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    if label == 'MSE1imppred':
        imputed_all_vals = pred_alt_chrimp[:,5].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,18].flatten().astype(float)
        imputed_chrimp_vals = pred_alt_chrimp[:,19].flatten().astype(float)
    elif label == 'MSE1impme':
        imputed_all_vals = me2d_alt_pred[:,19].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,5].flatten().astype(float)
        imputed_chrimp_vals = chrimp_alt_me2d[:,18].flatten().astype(float)
    else:
        imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
        imputed_me_vals = me2d_alt_pred[:,col_idx].flatten().astype(float)
        imputed_chrimp_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_impall, impall_yerrs, to_plot_impme, impme_yerrs, to_plot_chrimp, chrimp_yerrs = [], [], [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(pred_alt_chrimp[:,2] == str(assay_idx))
        to_plot_impall.append(imputed_all_vals[coord])
        to_plot_impme.append(imputed_me_vals[coord])
        to_plot_chrimp.append(imputed_chrimp_vals[coord])

    to_plot_impall = numpy.array(to_plot_impall, dtype=object)
    to_plot_impme = numpy.array(to_plot_impme, dtype=object)
    to_plot_chrimp = numpy.array(to_plot_chrimp, dtype=object)

    if label == 'MSEglobal':
        gmse_sort = numpy.argsort([numpy.mean(elt) for elt in to_plot_impall])
    ind = numpy.arange(0, len(sorted_assays)*4)
#    width = 0.18
    width = 0.25
    edgecol = 'k'
    edgewidth = 0.5

#    #main effects violins
#    vparts = axes[ax_coord].violinplot(to_plot_impme[gmse_sort], positions=ind[::4], showmeans=True, showextrema=True)
#    for pc in vparts['bodies']:
#        pc.set_color('violet')
##        pc.set_facecolor('violet')
##        pc.set_edgecolor('black')
#    #predictd violins
#    vparts = axes[ax_coord].violinplot(to_plot_impall[gmse_sort], positions=ind[1::4], showmeans=True, showextrema=True)
#    for pc in vparts['bodies']:
#        pc.set_color('blue')
#    #chromimpute violins
#    vparts = axes[ax_coord].violinplot(to_plot_chrimp[gmse_sort], positions=ind[2::4], showmeans=True, showextrema=True)
#    for pc in vparts['bodies']:
#        pc.set_color('red')

    c = 'violet'
    bparts = axes[ax_coord].boxplot(to_plot_impme[gmse_sort], positions=ind[::4], 
                                    boxprops=dict(facecolor=c, color='k'),
                                    capprops=dict(color='k'),
                                    whiskerprops=dict(color='k'),
                                    flierprops=dict(color='k', markerfacecolor='k', markeredgecolor='k'),
                                    medianprops=dict(color='gold', linewidth=2, solid_capstyle="butt"),
#                                    medianprops=dict(color='gold'),
                                    sym='.', patch_artist=True, widths=0.8)
    c = 'blue'
    bparts = axes[ax_coord].boxplot(to_plot_impall[gmse_sort], positions=ind[1::4], 
                                    boxprops=dict(facecolor=c, color='k'),
                                    capprops=dict(color='k'),
                                    whiskerprops=dict(color='k'),
                                    flierprops=dict(color='k', markerfacecolor='k', markeredgecolor='k'),
                                    medianprops=dict(color='gold', linewidth=2, solid_capstyle="butt"),
#                                    medianprops=dict(color='gold'),
                                    sym='.', patch_artist=True, widths=0.8)
    c = 'red'
    bparts = axes[ax_coord].boxplot(to_plot_chrimp[gmse_sort], positions=ind[2::4], 
                                    boxprops=dict(facecolor=c, color='k'),
                                    capprops=dict(color='k'),
                                    whiskerprops=dict(color='k'),
                                    flierprops=dict(color='k', markerfacecolor='k', markeredgecolor='k'),
                                    medianprops=dict(color='gold', linewidth=2, solid_capstyle="butt"),
#                                    medianprops=dict(color='gold'),
                                    sym='.', patch_artist=True, widths=0.8)

#    axes[ax_coord].set_xticks(ind + (1.5 * width))
    axes[ax_coord].set_xticks(ind[1::4])
    axes[ax_coord].set_xticklabels([sorted_assays[idx][0].replace('H2AZ', 'H2A.Z') for idx in gmse_sort], rotation=60, ha='right')

    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label.replace('asinh', ''), metric_desc[label][1].replace('with asinh data', '')), fontsize=20)
#    if ax_idx == axes.shape[1] - 1:
#        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
#        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Imputed Assay', fontsize=20)
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
    axes[ax_coord].set_xlim(-1,len(ind))
    
    
    ymin, ymax = (numpy.min([numpy.min(elt) for elt in itertools.chain(*[to_plot_impme, to_plot_impall, to_plot_chrimp])]),
                  numpy.max([numpy.max(elt) for elt in itertools.chain(*[to_plot_impme, to_plot_impall, to_plot_chrimp])]))
    if 'MSE' in label:
        axes[ax_coord].set_yscale('log')
        axes[ax_coord].set_ylim(ymin, ymax)
    if label == 'MSEglobal':
        axes[ax_coord].set_yticks([ymin, 0.1, ymax])
        axes[ax_coord].set_yticklabels(['{:.2e}'.format(ymin), '{:.2e}'.format(0.1), '{:.2e}'.format(ymax)])
#    axes[ax_coord].set_aspect('equal', 'datalim')

fig.tight_layout()
fig.savefig('../../predictd_figs/quality_measures_boxplots_pred_me_chrimp.pdf', bbox_inches='tight')


# ##Only one training experiment for CD3_Primary_Cells_Cord_BI cell type

# In[19]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

assays_present = set(list(itertools.chain(*[[tuple(data[i,(0,2)]) for i in range(data.shape[0])] 
                                     for data in (cd3_cord_h3k4me1, cd3_cord_h3k4me3)])))
sorted_assays = sorted(((elt[0].split('-')[-1], int(elt[1])) for elt in assays_present), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5),
                                           ('GWcorrasinh', 6),# ('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    imputed_h3k4me1_vals = cd3_cord_h3k4me1[:,col_idx].flatten().astype(float)
    imputed_h3k4me3_vals = cd3_cord_h3k4me3[:,col_idx].flatten().astype(float)
    imputed_h3k49me3_vals = cd3_cord_h3k4me3_h3k9me3[:,col_idx].flatten().astype(float)
    imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
    chrimp_plus_imputed_all_vals = pred_chrimp_avg[:,col_idx].flatten().astype(float)
    chrimp_metric_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_chrimp, to_plot_chrimp_plus_impall, to_plot_impall, to_plot_h3k4me1, to_plot_h3k4me3, to_plot_h3k49me3 = [], [], [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(numpy.logical_and(chrimp_alt_me2d[:,2] == str(assay_idx), chrimp_alt_me2d[:,1] == '23'))
        to_plot_chrimp.append(chrimp_metric_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(numpy.logical_and(pred_chrimp_avg[:,2] == str(assay_idx), pred_chrimp_avg[:,1] == '23'))
        to_plot_chrimp_plus_impall.append(chrimp_plus_imputed_all_vals[coord] if len(coord[0]) else numpy.nan)

        coord = numpy.where(numpy.logical_and(pred_alt_chrimp[:,2] == str(assay_idx), pred_alt_chrimp[:,1] == '23'))
        to_plot_impall.append(imputed_all_vals[coord] if len(coord[0]) else numpy.nan)

        coord = numpy.where(cd3_cord_h3k4me1[:,2] == str(assay_idx))
        to_plot_h3k4me1.append(imputed_h3k4me1_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(cd3_cord_h3k4me3[:,2] == str(assay_idx))
        to_plot_h3k4me3.append(imputed_h3k4me3_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(cd3_cord_h3k4me3_h3k9me3[:,2] == str(assay_idx))
        to_plot_h3k49me3.append(imputed_h3k49me3_vals[coord] if len(coord[0]) else numpy.nan)

    ind = numpy.arange(len(sorted_assays))
    width = 0.1
    edgecol = 'k'
    edgewidth = 0.5
    axes[ax_coord].bar(ind, to_plot_h3k4me1, width, color='orange', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me1-trained')
    axes[ax_coord].bar(ind + width, to_plot_h3k4me3, width, color='green', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me3-trained')
    axes[ax_coord].bar(ind + (2*width), to_plot_h3k49me3, width, color='yellow', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me3/H3K9me3-trained')
    axes[ax_coord].bar(ind + (3*width), to_plot_impall, width, color='blue', edgecolor=edgecol, linewidth=edgewidth, label='PREDICTD all')
    axes[ax_coord].bar(ind + (4*width), to_plot_chrimp_plus_impall, width, color='red', edgecolor=edgecol, linewidth=edgewidth, label='ChrImp-PREDICTD avg')
    axes[ax_coord].bar(ind + (5*width), to_plot_chrimp, width, color='purple', edgecolor=edgecol, linewidth=edgewidth, label='ChromImpute')
 
    axes[ax_coord].set_xticks(ind + (3 * width))
    axes[ax_coord].set_xticklabels([elt[0].replace('H2AZ', 'H2A.Z') for elt in sorted_assays], rotation=30, ha='right')
    
    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label, metric_desc[label][1]), fontsize=20)
    if ax_idx == axes.shape[1] - 1:
        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Assay Imputed for CD3 Primary Cells from Cord Blood')
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
#    axes[ax_coord].set_aspect('equal', 'datalim')

axes[3,1].set_axis_off()
axes[3,2].set_axis_off()
fig.tight_layout()
fig.savefig('../../predictd_figs/cd3_low_data_predictd_chrimp.pdf', bbox_inches='tight')


# In[20]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

assays_present = set(list(itertools.chain(*[[tuple(data[i,(0,2)]) for i in range(data.shape[0])] 
                                     for data in (cd3_cord_h3k4me1, cd3_cord_h3k4me3)])))
sorted_assays = sorted(((elt[0].split('-')[-1], int(elt[1])) for elt in assays_present), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5),
                                           ('GWcorrasinh', 6),# ('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    imputed_h3k4me1_vals = cd3_cord_h3k4me1[:,col_idx].flatten().astype(float)
    imputed_h3k4me3_vals = cd3_cord_h3k4me3[:,col_idx].flatten().astype(float)
    imputed_h3k49me3_vals = cd3_cord_h3k4me3_h3k9me3[:,col_idx].flatten().astype(float)
    imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
    chrimp_plus_imputed_all_vals = pred_chrimp_avg[:,col_idx].flatten().astype(float)
    chrimp_metric_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_chrimp, to_plot_chrimp_plus_impall, to_plot_impall, to_plot_h3k4me1, to_plot_h3k4me3, to_plot_h3k49me3 = [], [], [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(numpy.logical_and(chrimp_alt_me2d[:,2] == str(assay_idx), chrimp_alt_me2d[:,1] == '23'))
        to_plot_chrimp.append(chrimp_metric_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(numpy.logical_and(pred_chrimp_avg[:,2] == str(assay_idx), pred_chrimp_avg[:,1] == '23'))
        to_plot_chrimp_plus_impall.append(chrimp_plus_imputed_all_vals[coord] if len(coord[0]) else numpy.nan)

        coord = numpy.where(numpy.logical_and(pred_alt_chrimp[:,2] == str(assay_idx), pred_alt_chrimp[:,1] == '23'))
        to_plot_impall.append(imputed_all_vals[coord] if len(coord[0]) else numpy.nan)

        coord = numpy.where(cd3_cord_h3k4me1[:,2] == str(assay_idx))
        to_plot_h3k4me1.append(imputed_h3k4me1_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(cd3_cord_h3k4me3[:,2] == str(assay_idx))
        to_plot_h3k4me3.append(imputed_h3k4me3_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(cd3_cord_h3k4me3_h3k9me3[:,2] == str(assay_idx))
        to_plot_h3k49me3.append(imputed_h3k49me3_vals[coord] if len(coord[0]) else numpy.nan)

    ind = numpy.arange(len(sorted_assays))
    width = 0.15
    edgecol = 'k'
    edgewidth = 0.5
    axes[ax_coord].bar(ind, to_plot_h3k4me1, width, color='orange', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me1-trained')
    axes[ax_coord].bar(ind + width, to_plot_h3k4me3, width, color='green', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me3-trained')
    axes[ax_coord].bar(ind + (2*width), to_plot_h3k49me3, width, color='yellow', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me3/H3K9me3-trained')
    axes[ax_coord].bar(ind + (3*width), to_plot_impall, width, color='blue', edgecolor=edgecol, linewidth=edgewidth, label='PREDICTD all')
#    axes[ax_coord].bar(ind + (4*width), to_plot_chrimp_plus_impall, width, color='red', edgecolor=edgecol, linewidth=edgewidth, label='ChrImp-Imputed Avg')
#    axes[ax_coord].bar(ind + (5*width), to_plot_chrimp, width, color='purple', edgecolor=edgecol, linewidth=edgewidth, label='ChromImpute')
 
    axes[ax_coord].set_xticks(ind + (3 * width))
    axes[ax_coord].set_xticklabels([elt[0].replace('H2AZ', 'H2A.Z') for elt in sorted_assays], rotation=30, ha='right')
    
    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label, metric_desc[label][1]), fontsize=20)
    if ax_idx == axes.shape[1] - 1:
        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Assay Imputed for CD3 Primary Cells from Cord Blood')
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
#    axes[ax_coord].set_aspect('equal', 'datalim')

axes[3,1].set_axis_off()
axes[3,2].set_axis_off()
fig.tight_layout()
fig.savefig('../../predictd_figs/cd3_low_data_predictd.pdf', bbox_inches='tight')


# ##Imputation With Only H3K4me3 for more cell types

# ###GM12878_Lymphoblastoid

# In[21]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

assays_present = set(list(itertools.chain(*[[tuple(data[i,(0,2)]) for i in range(data.shape[0])] 
                                     for data in (gm12878_h3k4me3,)])))
sorted_assays = sorted(((elt[0].split('-')[-1], int(elt[1])) for elt in assays_present), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

cell_type_idx = '59'
cell_type_data = gm12878_h3k4me3
for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5),
                                           ('GWcorrasinh', 6),# ('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    imputed_h3k4me3_vals = cell_type_data[:,col_idx].flatten().astype(float)
    imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
    chrimp_plus_imputed_all_vals = pred_chrimp_avg[:,col_idx].flatten().astype(float)
    chrimp_metric_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_chrimp, to_plot_chrimp_plus_impall, to_plot_impall, to_plot_h3k4me3 = [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(numpy.logical_and(chrimp_alt_me2d[:,2] == str(assay_idx), chrimp_alt_me2d[:,1] == cell_type_idx))
        to_plot_chrimp.append(chrimp_metric_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(numpy.logical_and(pred_chrimp_avg[:,2] == str(assay_idx), pred_chrimp_avg[:,1] == cell_type_idx))
        to_plot_chrimp_plus_impall.append(chrimp_plus_imputed_all_vals[coord] if len(coord[0]) else numpy.nan)

        coord = numpy.where(numpy.logical_and(pred_alt_chrimp[:,2] == str(assay_idx), pred_alt_chrimp[:,1] == cell_type_idx))
        to_plot_impall.append(imputed_all_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(cell_type_data[:,2] == str(assay_idx))
        to_plot_h3k4me3.append(imputed_h3k4me3_vals[coord] if len(coord[0]) else numpy.nan)

    ind = numpy.arange(len(sorted_assays))
    width = 0.15
    edgecol = 'k'
    edgewidth = 0.5
    axes[ax_coord].bar(ind, to_plot_h3k4me3, width, color='green', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me3-trained')
    axes[ax_coord].bar(ind + width, to_plot_impall, width, color='blue', edgecolor=edgecol, linewidth=edgewidth, label='PREDICTD 5 test splits')
    axes[ax_coord].bar(ind + (2*width), to_plot_chrimp_plus_impall, width, color='red', edgecolor=edgecol, linewidth=edgewidth, label='ChrImp-PREDICTD avg')
    axes[ax_coord].bar(ind + (3*width), to_plot_chrimp, width, color='purple', edgecolor=edgecol, linewidth=edgewidth, label='ChromImpute')
 
    axes[ax_coord].set_xticks(ind + (3 * width))
    axes[ax_coord].set_xticklabels([elt[0].replace('H2AZ', 'H2A.Z') for elt in sorted_assays], rotation=30, ha='right')
    
    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label, metric_desc[label][1]), fontsize=20)
    if ax_idx == axes.shape[1] - 1:
        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Assay Imputed for CD3 Primary Cells from Cord Blood')
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
#    axes[ax_coord].set_aspect('equal', 'datalim')

axes[3,1].set_axis_off()
axes[3,2].set_axis_off()
fig.tight_layout()
fig.savefig('../../predictd_figs/h3k4me3_only_gm12878_lymphoblastoid.pdf', bbox_inches='tight')


# ###Fetal_Muscle_Trunk

# In[22]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

cell_type_idx = '55'
cell_type_data = fetal_muscle_h3k4me3

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

assays_present = set(list(itertools.chain(*[[tuple(data[i,(0,2)]) for i in range(data.shape[0])] 
                                     for data in (cell_type_data,)])))
sorted_assays = sorted(((elt[0].split('-')[-1], int(elt[1])) for elt in assays_present), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5),
                                           ('GWcorrasinh', 6),# ('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    imputed_h3k4me3_vals = cell_type_data[:,col_idx].flatten().astype(float)
    imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
    chrimp_plus_imputed_all_vals = pred_chrimp_avg[:,col_idx].flatten().astype(float)
    chrimp_metric_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_chrimp, to_plot_chrimp_plus_impall, to_plot_impall, to_plot_h3k4me3 = [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(numpy.logical_and(chrimp_alt_me2d[:,2] == str(assay_idx), chrimp_alt_me2d[:,1] == cell_type_idx))
        to_plot_chrimp.append(chrimp_metric_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(numpy.logical_and(pred_chrimp_avg[:,2] == str(assay_idx), pred_chrimp_avg[:,1] == cell_type_idx))
        to_plot_chrimp_plus_impall.append(chrimp_plus_imputed_all_vals[coord] if len(coord[0]) else numpy.nan)

        coord = numpy.where(numpy.logical_and(pred_alt_chrimp[:,2] == str(assay_idx), pred_alt_chrimp[:,1] == cell_type_idx))
        to_plot_impall.append(imputed_all_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(cell_type_data[:,2] == str(assay_idx))
        to_plot_h3k4me3.append(imputed_h3k4me3_vals[coord] if len(coord[0]) else numpy.nan)

    ind = numpy.arange(len(sorted_assays))
    width = 0.15
    edgecol = 'k'
    edgewidth = 0.5
    axes[ax_coord].bar(ind, to_plot_h3k4me3, width, color='green', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me3-trained')
    axes[ax_coord].bar(ind + width, to_plot_impall, width, color='blue', edgecolor=edgecol, linewidth=edgewidth, label='PREDICTD 5 test splits')
    axes[ax_coord].bar(ind + (2*width), to_plot_chrimp_plus_impall, width, color='red', edgecolor=edgecol, linewidth=edgewidth, label='ChrImp-PREDICTD avg')
    axes[ax_coord].bar(ind + (3*width), to_plot_chrimp, width, color='purple', edgecolor=edgecol, linewidth=edgewidth, label='ChromImpute')
 
    axes[ax_coord].set_xticks(ind + (3 * width))
    axes[ax_coord].set_xticklabels([elt[0].replace('H2AZ', 'H2A.Z') for elt in sorted_assays], rotation=30, ha='right')
    
    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label, metric_desc[label][1]), fontsize=20)
    if ax_idx == axes.shape[1] - 1:
        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Assay Imputed for CD3 Primary Cells from Cord Blood')
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
#    axes[ax_coord].set_aspect('equal', 'datalim')

axes[3,1].set_axis_off()
axes[3,2].set_axis_off()
fig.tight_layout()
fig.savefig('../../predictd_figs/h3k4me3_only_fetal_muscle_trunk.pdf', bbox_inches='tight')


# ###Brain_Anterior_Caudate

# In[23]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

cell_type_idx = '8'
cell_type_data = brain_anterior_caudate_h3k4me3

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

assays_present = set(list(itertools.chain(*[[tuple(data[i,(0,2)]) for i in range(data.shape[0])] 
                                     for data in (cell_type_data,)])))
sorted_assays = sorted(((elt[0].split('-')[-1], int(elt[1])) for elt in assays_present), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5),
                                           ('GWcorrasinh', 6),# ('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    imputed_h3k4me3_vals = cell_type_data[:,col_idx].flatten().astype(float)
    imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
    chrimp_plus_imputed_all_vals = pred_chrimp_avg[:,col_idx].flatten().astype(float)
    chrimp_metric_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_chrimp, to_plot_chrimp_plus_impall, to_plot_impall, to_plot_h3k4me3 = [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(numpy.logical_and(chrimp_alt_me2d[:,2] == str(assay_idx), chrimp_alt_me2d[:,1] == cell_type_idx))
        to_plot_chrimp.append(chrimp_metric_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(numpy.logical_and(pred_chrimp_avg[:,2] == str(assay_idx), pred_chrimp_avg[:,1] == cell_type_idx))
        to_plot_chrimp_plus_impall.append(chrimp_plus_imputed_all_vals[coord] if len(coord[0]) else numpy.nan)

        coord = numpy.where(numpy.logical_and(pred_alt_chrimp[:,2] == str(assay_idx), pred_alt_chrimp[:,1] == cell_type_idx))
        to_plot_impall.append(imputed_all_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(cell_type_data[:,2] == str(assay_idx))
        to_plot_h3k4me3.append(imputed_h3k4me3_vals[coord] if len(coord[0]) else numpy.nan)

    ind = numpy.arange(len(sorted_assays))
    width = 0.15
    edgecol = 'k'
    edgewidth = 0.5
    axes[ax_coord].bar(ind, to_plot_h3k4me3, width, color='green', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me3-trained')
    axes[ax_coord].bar(ind + width, to_plot_impall, width, color='blue', edgecolor=edgecol, linewidth=edgewidth, label='PREDICTD 5 test splits')
    axes[ax_coord].bar(ind + (2*width), to_plot_chrimp_plus_impall, width, color='red', edgecolor=edgecol, linewidth=edgewidth, label='ChrImp-PREDICTD avg')
    axes[ax_coord].bar(ind + (3*width), to_plot_chrimp, width, color='purple', edgecolor=edgecol, linewidth=edgewidth, label='ChromImpute')
 
    axes[ax_coord].set_xticks(ind + (3 * width))
    axes[ax_coord].set_xticklabels([elt[0].replace('H2AZ', 'H2A.Z') for elt in sorted_assays], rotation=30, ha='right')
    
    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label, metric_desc[label][1]), fontsize=20)
    if ax_idx == axes.shape[1] - 1:
        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Assay Imputed for CD3 Primary Cells from Cord Blood')
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
#    axes[ax_coord].set_aspect('equal', 'datalim')

axes[3,1].set_axis_off()
axes[3,2].set_axis_off()
fig.tight_layout()
fig.savefig('../../predictd_figs/h3k4me3_only_brain_anterior_caudate.pdf', bbox_inches='tight')


# ###Lung

# In[24]:

pyplot.rcParams.update({'xtick.labelsize':16,
                        'ytick.labelsize':16})

cell_type_idx = '81'
cell_type_data = lung_h3k4me3

fig, axes = pyplot.subplots(nrows=4, ncols=3, figsize=(24,32))

assays_present = set(list(itertools.chain(*[[tuple(data[i,(0,2)]) for i in range(data.shape[0])] 
                                     for data in (cell_type_data,)])))
sorted_assays = sorted(((elt[0].split('-')[-1], int(elt[1])) for elt in assays_present), key=lambda x:x[1])

colors = pyplot.cm.Set1(numpy.linspace(0, 1, len(sorted_assays)))

metric_desc = {'MSEglobal': ('Mean squared error', 'Genome-wide mean squared error'),
               'MSE1obs': ('Mean squared error', 'Mean squared error at top 1% obs win'),
               'MSE1imp': ('Mean squared error', 'Mean squared error at top 1% imp win'),
               'MSE1imppred': ('Mean squared error', 'Mean squared error at top 1% PREDICTD win'),
               'MSE1impme': ('Mean squared error', 'Mean squared error at top 1% Main Effects win'),
               'MSE1impchrimp': ('Mean squared error', 'Mean squared error at top 1% ChromImpute win'),
               'GWcorrasinh': ('Pearson\'s r', 'Genome-wide Pearson correlation of asinh data'),
               'GWspcorr': ('Spearman\'s r', 'Genome-wide Spearman correlation'),
               'GWcorr': ('Pearson\'s r', 'Genome-wide Pearson correlation'),
               'Match1': ('Fraction overlap', 'Imp & obs overlap in top 1% signal windows'),
               'Catch1obs': ('Fraction overlap', 'Fraction of top 1% obs in top 5% imp'),
               'Catch1imp': ('Fraction overlap', 'Fraction of top 1% imp in top 5% obs'),
               'AucObs1': ('ROC AUC', 'Recovery of top 1% obs from all imp'),
               'AucImp1': ('ROC AUC', 'Recovery of top 1% imp from all obs'),
               'CatchPeakObs': ('ROC AUC', 'Recovery of obs peak calls from all imp')}

for ax_idx, (label, col_idx) in enumerate([('MSEglobal', 3), ('MSE1obs', 4), ('MSE1imp', 5),
                                           ('GWcorrasinh', 6),# ('GWspcorr', 8), ('GWcorr', 10), 
                                           ('Match1', 12), ('Catch1obs', 13), ('Catch1imp', 14), 
                                           ('AucObs1', 15), ('AucImp1', 16), ('CatchPeakObs', 17)]):
    maxval, minval = (0,1000)
    ax_coord = (ax_idx/axes.shape[1], ax_idx % axes.shape[1])
    imputed_h3k4me3_vals = cell_type_data[:,col_idx].flatten().astype(float)
    imputed_all_vals = pred_alt_chrimp[:,col_idx].flatten().astype(float)
    chrimp_plus_imputed_all_vals = pred_chrimp_avg[:,col_idx].flatten().astype(float)
    chrimp_metric_vals = chrimp_alt_me2d[:,col_idx].flatten().astype(float)
    to_plot_chrimp, to_plot_chrimp_plus_impall, to_plot_impall, to_plot_h3k4me3 = [], [], [], []
    for assay_name, assay_idx in sorted_assays:
        coord = numpy.where(numpy.logical_and(chrimp_alt_me2d[:,2] == str(assay_idx), chrimp_alt_me2d[:,1] == cell_type_idx))
        to_plot_chrimp.append(chrimp_metric_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(numpy.logical_and(pred_chrimp_avg[:,2] == str(assay_idx), pred_chrimp_avg[:,1] == cell_type_idx))
        to_plot_chrimp_plus_impall.append(chrimp_plus_imputed_all_vals[coord] if len(coord[0]) else numpy.nan)

        coord = numpy.where(numpy.logical_and(pred_alt_chrimp[:,2] == str(assay_idx), pred_alt_chrimp[:,1] == cell_type_idx))
        to_plot_impall.append(imputed_all_vals[coord] if len(coord[0]) else numpy.nan)
        
        coord = numpy.where(cell_type_data[:,2] == str(assay_idx))
        to_plot_h3k4me3.append(imputed_h3k4me3_vals[coord] if len(coord[0]) else numpy.nan)

    ind = numpy.arange(len(sorted_assays))
    width = 0.15
    edgecol = 'k'
    edgewidth = 0.5
    axes[ax_coord].bar(ind, to_plot_h3k4me3, width, color='green', edgecolor=edgecol, linewidth=edgewidth, label='H3K4me3-trained')
    axes[ax_coord].bar(ind + width, to_plot_impall, width, color='blue', edgecolor=edgecol, linewidth=edgewidth, label='PREDICTD 5 test splits')
    axes[ax_coord].bar(ind + (2*width), to_plot_chrimp_plus_impall, width, color='red', edgecolor=edgecol, linewidth=edgewidth, label='ChrImp-PREDICTD avg')
    axes[ax_coord].bar(ind + (3*width), to_plot_chrimp, width, color='purple', edgecolor=edgecol, linewidth=edgewidth, label='ChromImpute')
 
    axes[ax_coord].set_xticks(ind + (3 * width))
    axes[ax_coord].set_xticklabels([elt[0].replace('H2AZ', 'H2A.Z') for elt in sorted_assays], rotation=30, ha='right')
    
    axes[ax_coord].set_title('\n{!s}\n{!s}\n'.format(label, metric_desc[label][1]), fontsize=20)
    if ax_idx == axes.shape[1] - 1:
        legend = axes[ax_coord].legend(loc='upper left', bbox_to_anchor=[1, 1], fontsize=20)
        legend.get_frame().set_facecolor('#FFFFFF')
#    axes[ax_coord].set_xlabel('Assay Imputed for CD3 Primary Cells from Cord Blood')
    axes[ax_coord].set_ylabel(metric_desc[label][0], fontsize=20)
#    axes[ax_coord].set_aspect('equal', 'datalim')

axes[3,1].set_axis_off()
axes[3,2].set_axis_off()
fig.tight_layout()
fig.savefig('../../predictd_figs/h3k4me3_only_lung.pdf', bbox_inches='tight')


# ###The CD3_Primary_Cells_Cord_BI cell type data sets are distributed pretty evenly across the 5 test folds:
# 
# | Cell Type | Assay | Test Fold |
# | : --- | : --- | : --- : |
# | CD3_Primary_Cells_Cord_BI | H3K27me3 | 2 |
# | CD3_Primary_Cells_Cord_BI | H3K9me3 | 2 |
# | CD3_Primary_Cells_Cord_BI | DNase | 0 |
# | CD3_Primary_Cells_Cord_BI | H3K4me3 | 4 |
# | CD3_Primary_Cells_Cord_BI | H3K36me3 | 1 |
# | CD3_Primary_Cells_Cord_BI | H3K4me1 | 0 |

# In[24]:



