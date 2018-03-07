
'''
Once a model is trained, its cell type and assay parameters can be applied across the genome to impute whole genome tracks. This is accomplished by reading in the assembled bedgraph data from `assemble_new_ct_datasets.py` and `upload_genome_file_in_parts.py`, training the corresponding genome parameters for each line of bedgraph data, and printing out the imputed values. This script can also average the results of multiple PREDICTD models and print the averaged values as the output.
'''

import argparse
import itertools
import numpy
import os
from pyspark import SparkContext, StorageLevel
import scipy.sparse as sps
import smart_open
import sys

sys.path.append(os.path.dirname(__file__))
import s3_library
import predictd_lib as pl
import assembled_data_to_rdd as adtr

def avg_impute(gtotal_elt, second_order_params, all_coords, num_valids, test_coord_list, missing_coords_groups, coords_to_impute=None, remove_negs=True, master_seed=1):
    #find coordinates of observed data and coordinates of missing data
    if len(gtotal_elt) == 2:
        orig_data_elt = gtotal_elt[1][0]
    else:
        orig_data_elt = gtotal_elt[1]
    #now, impute from each model and sum the corresponding imputed values
    final_impute = numpy.zeros(orig_data_elt.shape, dtype=float)
#    if final_impute.shape != (127,24):
#        raise Exception(gtotal_elt[0])
    impute_sum = final_impute.copy()
    rs = numpy.random.RandomState((master_seed * gtotal_elt[0][1]) % 4294967295)
    missing_coords_idx = numpy.arange(len(missing_coords_groups.value))
    rs.shuffle(missing_coords_idx)
    for idx,((tidx, vidx), (z_mat, z_h, ac_bias, ct_assay, ct_assay_bias, gmean, subsets_flat)) in enumerate(second_order_params.value):
        data_elt = orig_data_elt.copy()
        data_elt.data -= gmean
        #res is: (data, genome, genome_bias)
        gidx, res = pl.second_order_genome_updates((gtotal_elt[0], (data_elt, None, None)), numpy.where(~subsets_flat[pl.SUBSET_TRAIN]), z_mat, z_h, ac_bias)
#        imputed = pl._compute_imputed2_helper((gidx, res), ct_assay, ct_assay_bias, gmean, coords=coords_to_impute.value if coords_to_impute is not None else None)[1].toarray()
        imputed = pl._compute_imputed2_helper((gidx, res), ct_assay, ct_assay_bias, gmean, coords=None, enforce_nonegs=remove_negs)[1]
        test_coords = test_coord_list.value[idx]
        if test_coords:
            final_impute[test_coords] += imputed[test_coords]

        if missing_coords_groups is not None:
            non_test_coords = missing_coords_groups.value[missing_coords_idx[idx]]
            if non_test_coords is not None:
                final_impute[non_test_coords] += imputed[non_test_coords]

    #finish computing the average
    final_impute[all_coords.value] /= num_valids
    if numpy.sum(final_impute == 0)/float(numpy.multiply(*final_impute.shape)) < 0.2:
        return (gidx, final_impute)
    else:
        return (gidx, sps.csr_matrix(final_impute))
#    csr_to_return = sps.csr_matrix(final_impute)
#    if remove_negs is True:
#        return (gidx, pl._remove_negs_csr(csr_to_return))
#    else:
#        return (gidx, csr_to_return)

def prep_ctassays(hyperparams_path, subsets, valid_ct_model_coords, valid_assay_model_coords, ri2=2.9):
    bucket_txt, key_txt = s3_library.parse_s3_url(hyperparams_path)
    ct_key_txt = os.path.join(os.path.dirname(key_txt), 'ct_factors.pickle')
    ct, ct_bias = s3_library.get_pickle_s3(bucket_txt, ct_key_txt)
    print(ct.shape)
    print(ct_bias.shape)
    ct, ct_bias = ct[list(valid_ct_model_coords),:], ct_bias[list(valid_ct_model_coords)]

    assay_key_txt = os.path.join(os.path.dirname(key_txt), 'assay_factors.pickle')
    assay, assay_bias = s3_library.get_pickle_s3(bucket_txt, assay_key_txt)
    assay, assay_bias = assay[list(valid_assay_model_coords),:], assay_bias[list(valid_assay_model_coords)]

    gmean = s3_library.get_pickle_s3(bucket_txt, os.path.join(os.path.dirname(key_txt), 'gmean.pickle'))

    all_valid_coords = list(zip(*itertools.product(valid_ct_model_coords, valid_assay_model_coords)))
    subsets_flat = [subsets[i][all_valid_coords].flatten() for i in range(len(subsets))]
    subsets_flat_coords = numpy.where(~subsets_flat[pl.SUBSET_TRAIN])
    ac_bias = numpy.add.outer(ct_bias, assay_bias).flatten()[subsets_flat_coords]
    ct_z = numpy.hstack([ct, numpy.ones((ct.shape[0], 1))])
    assay_z = numpy.hstack([assay, numpy.ones((assay.shape[0], 1))])
    z_mat = numpy.vstack([numpy.outer(ct_z[:,idx], assay_z[:,idx]).flatten()[subsets_flat_coords] for idx in xrange(ct_z.shape[1])])
    reg_coef_add = numpy.ones(z_mat.shape[0])
    reg_coef_add[-1] = 0
    reg_coef_add = numpy.diag(reg_coef_add * ri2)
    z_h = numpy.linalg.inv(numpy.dot(z_mat, z_mat.T) + reg_coef_add)

    ct_assay = numpy.vstack([numpy.outer(ct[:,idx], assay[:,idx]).flatten() for idx in xrange(ct.shape[1])])
    ct_assay_bias = numpy.add.outer(ct_bias, assay_bias).flatten()
    return (z_mat, z_h, ac_bias, ct_assay, ct_assay_bias, gmean, subsets_flat)

def _calculate_averaging_scheme(data_shape, second_order_params, master_seed, coords_to_impute=None, addl_agg_data_shape=None, random_state=None, ignore_test_sets=False):
    '''This function takes in the subsets objects for the models being used for imputation,
    plus the coordinates of the tensor requested for imputation (if any), and computes the
    coordinates of present and missing data sets that were requested for imputation. The
    present experiments are imputed just from models that had those experiments in their
    test sets, and the missing experiments are imputed from a random subset of models with
    a size equal to the number of validation sets. This means that missing and present 
    experiments are imputed by averaging the same number of models. This function generates
    groups for the missing experiments that include the coordinates for each missing experiment
    in the same number of groups as the number of validation sets, and it extracts the test
    set coordinates for each model. The missing experiment groups and the test set coordinates
    can then be passed into avg_impute along with second_order_params to directly do the
    imputation.
    '''
    if addl_agg_data_shape is not None:
        #assume the additional data is adding in the cell type dimension, not the assay dimension
        data_shape = (data_shape[0] + addl_agg_data_shape[0], data_shape[1])
    subsets = second_order_params[0][1][6]
    present_coords = numpy.logical_or(~subsets[pl.SUBSET_TRAIN].reshape(data_shape),
                                      numpy.logical_or(~subsets[pl.SUBSET_VALID].reshape(data_shape),
                                                       ~subsets[pl.SUBSET_TEST].reshape(data_shape)))
    present_coords = set(zip(*numpy.where(present_coords)))
    #if specific coordinates have been requested, restrict the coordinate sets appropriately
    if coords_to_impute is not None:
        missing_coords = set(zip(*coords_to_impute)) - present_coords
        present_coords &= set(zip(*coords_to_impute))
    #otherwise, just parse all possible coordinates as present/missing
    else:
        missing_coords = numpy.ones(data_shape)
        missing_coords[zip(*present_coords)] = 0
        missing_coords = set(zip(*numpy.where(missing_coords)))
    all_coords = zip(*(present_coords | missing_coords))
    missing_coords = sorted(missing_coords)

    #generate averaging scheme so that all imputed values are the average
    #of the same number of models and observed values are only averaged from 
    #the proper test sets (assumes that each test set has the same number of validation sets)
    num_valids = len([elt for elt in second_order_params if elt[0][0] == second_order_params[0][0][0]])
    num_tests = len(second_order_params)/num_valids
    if missing_coords:
        #split missing_coords into a group for each test set so that
        #each can be added from num_valids models
        groupsize = (len(missing_coords)/num_tests) + 1
        print('Num missing coords: {!s}'.format(len(missing_coords)))
        print('Missing coords groupsize: {!s}'.format(groupsize))
        rs = random_state if random_state is not None else numpy.random.RandomState(master_seed % 4294967295)
        if groupsize:
            rs.shuffle(missing_coords)
            missing_coords_groups = [zip(*missing_coords[i:i+groupsize])
                                     for i in xrange(0, len(missing_coords), groupsize)] * num_valids
        else:
            missing_coords_groups = [zip(*missing_coords)] * num_valids
        #top up the number of groups if the number of missing coordinates isn't great enough to spread
        #across all test sets
        if len(missing_coords_groups) < len(second_order_params):
            print('Adding {!s} dummy missing groups to match up with number of models.'.format(len(second_order_params) - len(missing_coords_groups)))
            missing_coords_groups += [None] * (len(second_order_params) - len(missing_coords_groups))
#            print(len(missing_coords_groups))
#        sys.exit()
        rs.shuffle(missing_coords_groups)
    else:
        missing_coords_groups = None
    #get the test set coords for each model and filter them to match the 
    #present_coords from coords_to_impute
    test_coords_list = []
    for elt in second_order_params:
        subsets = elt[1][6]
#        print(elt[0], pl.SUBSET_TEST, numpy.sum(~subsets[0]), numpy.sum(~subsets[1]), numpy.sum(~subsets[2]))
        #this option is rarely used; only in cases where one wants to print imputed versions of all tracks
        #regardless of test/train sets, for example in the imputing new cell type demo
        if ignore_test_sets is True:
            test_coords_list.append(zip(*present_coords))
        else:
            test_coords = zip(*numpy.where(~subsets[pl.SUBSET_TEST].reshape(data_shape)))
            test_coords_list.append(zip(*[elt for elt in test_coords if elt in present_coords]))
    return all_coords, num_valids, test_coords_list, missing_coords_groups

def main(args):
    #read in metadata and locations of data files
    print('Loading Data')
    agg_data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.agg_data_idx))
    agg_coord_order = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.agg_coord_order))
    if args.agg_coord_order == 's3://predictd/Observed/training_data.alldata-parts.column_coords.pickle':
        correction = s3_library.get_pickle_s3('predictd', 'Observed/training_data.alldata-parts.data_idx_coord_to_alldata_coord_map.pickle')
        correction = {v:k for k,v in correction.items()}
        agg_coord_order = [correction[elt] for elt in agg_coord_order]
    agg_ct_coord_map = {elt[0]:elt[-1][0] for elt in agg_data_idx.values()}
    agg_assay_coord_map = {elt[1]:elt[-1][1] for elt in agg_data_idx.values()}

    if args.agg_parts:
        bucket_txt, key_txt = s3_library.parse_s3_url(args.agg_parts)
        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(key_txt, '*.part*.txt.gz'))
    else:
        bucket_txt, key_txt = s3_library.parse_s3_url(args.agg_coord_order)
        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(os.path.dirname(key_txt), '*.part*.txt.gz'))
        
    agg_data_parts = sorted(['s3://{!s}/{!s}'.format(bucket_txt, elt.name) for elt in parts_keys],
                            key=lambda x: int(os.path.basename(x).split('.')[1][4:]))

    if args.addl_agg_coord_order:
        addl_agg_data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.addl_agg_data_idx))
        bucket_txt, key_txt = s3_library.parse_s3_url(args.addl_agg_coord_order)
        addl_agg_coord_order = s3_library.get_pickle_s3(bucket_txt, key_txt)
        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(os.path.dirname(key_txt), '*.part*.txt.gz'))
        addl_agg_data_parts = sorted(['s3://{!s}/{!s}'.format(bucket_txt, elt.name) for elt in parts_keys],
                                     key=lambda x: int(os.path.basename(x).split('.')[1][4:]))
        
        if len(agg_data_parts) != len(addl_agg_data_parts):
            print(args.agg_parts, len(agg_data_parts), os.path.join(os.path.dirname(key_txt), '*.part*.txt.gz'), len(addl_agg_data_parts))
            raise Exception('agg_data and addl_agg_data directories must contain the same number of part files.')
        agg_data_parts = list(zip(agg_data_parts, addl_agg_data_parts))

    #read in models to use
    print('Loading Models')
    model_data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.model_data_idx))
    model_ctet_coord_map = {tuple(elt[:2]):elt[-1] for elt in model_data_idx}
    model_ct_coord_map = {elt[0]:elt[-1][0] for elt in model_data_idx.values()}
    model_assay_coord_map = {elt[1]:elt[-1][1] for elt in model_data_idx.values()}

    #only take requested ct/assays that are present in the model/data overlap set
    if args.cts_to_impute and args.assays_to_impute:
        requested_ct = set(args.cts_to_impute.split(',')) & set(model_ct_coord_map)
        requested_assay = set(args.assays_to_impute.split(',')) & set(model_assay_coord_map)
        if not requested_ct or not requested_assay:
            raise Exception('No requested ct or assay present in model.')
        requested_ctassay = list(itertools.product(sorted(requested_ct), sorted(requested_assay)))
    elif args.cts_to_impute:
        requested_ct = set(args.cts_to_impute.split(',')) & set(model_ct_coord_map)
        if not requested_ct:
            raise Exception('No requested ct present in data and model overlap.')
        requested_ctassay = list(itertools.product(sorted(requested_ct), sorted(model_assay_coord_map.keys())))
    elif args.assays_to_impute:
        requested_assay = set(args.assays_to_impute.split(',')) & set(model_assay_coord_map)
        if not requested_assay:
            raise Exception('No requested assay present in data and model overlap.')
        requested_ctassay = list(itertools.product(sorted(model_ct_coord_map.keys()), sorted(requested_assay)))
    else:
        requested_ctassay = list(itertools.product(sorted(model_ct_coord_map.keys()), 
                                                   sorted(model_assay_coord_map.keys())))
    requested_coords = [numpy.array(e2) for e2 in zip(*[(model_ct_coord_map[elt[0]], model_assay_coord_map[elt[1]]) 
                                  for elt in requested_ctassay])]

    #ensure that there is training data for the overlapping ct/assays
    valid_ct, valid_ct_model_coords = zip(*sorted(model_ct_coord_map.items(), key=lambda x:x[1]))
    print(len(valid_ct_model_coords))
    print(valid_ct_model_coords[:5])
    valid_assay, valid_assay_model_coords = zip(*sorted(model_assay_coord_map.items(), key=lambda x:x[1]))
    print(len(valid_assay_model_coords))
    print(valid_assay_model_coords[:5])
    valid_ctassay = list(itertools.product(valid_ct, valid_assay))

    bucket_txt, key_glob_txt = s3_library.parse_s3_url(args.model_url_glob)
    if not key_glob_txt.endswith('hyperparameters.pickle'):
        key_glob_txt = os.path.join(os.path.dirname(key_glob_txt), 'hyperparameters.pickle')
    hyperparam_paths = sorted([elt.name for elt in s3_library.glob_keys(bucket_txt, key_glob_txt)])
    print(key_glob_txt, hyperparam_paths)
    second_order_params = []
    for path in hyperparam_paths:
        print(path)
        hyperparams = s3_library.get_pickle_s3(bucket_txt, path)
        subsets = hyperparams['subsets']
        path_url = 's3://{!s}/{!s}'.format(bucket_txt, path)
        second_order_params.append(((hyperparams['args'].fold_idx, hyperparams['args'].valid_fold_idx), prep_ctassays(path_url, subsets, valid_ct_model_coords, valid_assay_model_coords, ri2=hyperparams['args'].ri2)))

    #impute data for all models and average
    print('Imputing and generating bedgraphs')
    assay_list = valid_assay
    ct_list = valid_ct
    tmpdir = args.tmpdir
    agg_data_shape = tuple(numpy.max(numpy.array([elt[-1] for elt in agg_data_idx.values()], dtype=int), axis=0) + 1)
    print('agg_data_shape: {!s}'.format(agg_data_shape))
    addl_agg_data_shape = None
    if args.addl_agg_coord_order:
        addl_agg_data_shape = tuple(numpy.max(numpy.array([elt[-1] for elt in addl_agg_data_idx.values()], dtype=int), axis=0) + 1)
        print('addl_agg_data_shape: {!s}'.format(addl_agg_data_shape))

    #Compute the averaging scheme for missing data
    rs = numpy.random.RandomState(args.random_seed % 4294967295)
#    print(second_order_params)
    all_coords, num_valids, test_coord_list, missing_coords_groups = _calculate_averaging_scheme(agg_data_shape, second_order_params, args.random_seed + 1, coords_to_impute=requested_coords, addl_agg_data_shape=addl_agg_data_shape, random_state=rs, ignore_test_sets=args.ignore_test_sets)

#    print(len(all_coords[0]))
#    print(num_valids)
#    print(len(test_coord_list))
#    print(len(test_coord_list[0]))
##    print(len(test_coord_list[0][0]))
#    print([(i, second_order_params[i][0]) for i in range(len(second_order_params)) if len(test_coord_list[i]) and (63, 11) in zip(*test_coord_list[i])])
#    print(len(missing_coords_groups))

    bdg_urls = []
    out_bucket, out_root = s3_library.parse_s3_url(args.out_root_url)
    for idx in xrange(0, len(agg_data_parts), args.parts_at_once):
        #start SparkContext
        sc = SparkContext(appName='impute_data_gen_bedgraph',
                          pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                                   pl.__file__.replace('.pyc', '.py'),
                                   adtr.__file__.replace('.pyc', '.py'),
                                   os.path.join(os.path.dirname(adtr.__file__), 'read_file_lines.py')])
        pl.sc = sc
        adtr.pl = pl

        #make read only variables for shared data
        all_coords_bc = sc.broadcast(all_coords)
        test_coord_list_bc = sc.broadcast(test_coord_list)
        missing_coords_groups_bc = sc.broadcast(missing_coords_groups)
        second_order_params_bc = sc.broadcast(second_order_params)
        requested_coords_bc = sc.broadcast(requested_coords)

        bdg_path = os.path.join(tmpdir, '{{0!s}}_{{1!s}}/{{0!s}}_{{1!s}}.{:05d}.{{2!s}}.txt'.format(idx))

        agg_mat_coords = list(zip(*agg_coord_order))
        if addl_agg_data_shape is None:
            if args.rdd_sort is True:
#THE BELOW LINE IS FOR DEBUGGING
#                imputed_part_tmp = sc.parallelize(['s3://encodeimputation-alldata/25bp/alldata-parts/alldata.part5.txt.gz'], numSlices=1)\
                imputed_part_tmp = sc.parallelize(agg_data_parts[idx:idx+args.parts_at_once], numSlices=args.parts_at_once)\
                                     .flatMap(lambda x: adtr.read_in_part(x, agg_data_shape, agg_mat_coords, coords_bed=args.coords_bed, tmpdir=tmpdir)).repartition(500).map(lambda x: adtr.line_to_data(x, agg_data_shape, agg_mat_coords, arcsinh=args.arcsinh)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
                imputed_part_tmp.count()
                imputed_part = imputed_part_tmp.sortByKey(numPartitions=500)\
                                               .map(lambda (x,y): avg_impute((x,y,None,None), second_order_params_bc, all_coords_bc, num_valids, test_coord_list_bc, missing_coords_groups_bc, coords_to_impute=requested_coords_bc, master_seed=args.random_seed))#\
#                                               .persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
#It would probably be better to materialize the RDD here.
#                imputed_part.count()
#                imputed_part_tmp.unpersist()
#                del(imputed_part_tmp)
            else:
                raise Exception('This code section needs updating')
                imputed_part = sc.parallelize(agg_data_parts[idx:idx+args.parts_at_once], numSlices=args.parts_at_once)\
                                 .flatMap(lambda x: adtr.read_in_part(x, agg_data_shape, agg_mat_coords))\
                                 .repartition(200)\
                                 .map(lambda (x,y): avg_impute((x,y,None,None), second_order_params))\
                                 .persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
                imputed_part.count()
        else:
#            print('reading in and combining data')
            addl_agg_mat_coords = list(zip(*addl_agg_coord_order))
#            print(addl_agg_mat_coords)
#            sys.exit()
            addl_agg_to_agg_idx = pl._make_rdd2_to_rdd1_idx(agg_data_idx, addl_agg_data_idx)
            if args.rdd_sort is True:
                imputed_part_tmp = sc.parallelize(agg_data_parts[idx:idx+args.parts_at_once], numSlices=args.parts_at_once)\
                                     .flatMap(lambda (x, y): zip(adtr.read_in_part(x, agg_data_shape, agg_mat_coords, coords_bed=args.coords_bed, tmpdir=tmpdir), adtr.read_in_part(y, addl_agg_data_shape, addl_agg_mat_coords, coords_bed=args.coords_bed, tmpdir=tmpdir)))\
                                     .repartition(500)\
                                     .map(lambda (x, y): (adtr.line_to_data(x, agg_data_shape, agg_mat_coords, arcsinh=args.arcsinh), adtr.line_to_data(y, addl_agg_data_shape, addl_agg_mat_coords, arcsinh=args.arcsinh)))\
                                     .map(lambda ((idx1,x), (idx2,y)): (idx1, pl._merge_rdds_helper(x,y,addl_agg_to_agg_idx))).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
                imputed_part_tmp.count()
                imputed_part = imputed_part_tmp.sortByKey(numPartitions=500)\
                                               .map(lambda (x,y): avg_impute((x,y,None,None), second_order_params_bc, all_coords_bc, num_valids, test_coord_list_bc, missing_coords_groups_bc, coords_to_impute=requested_coords_bc, master_seed=args.random_seed))#\
#                                 .persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
            else:
                raise Exception('This code section needs updating')
                imputed_part = sc.parallelize(agg_data_parts[idx:idx+args.parts_at_once], numSlices=args.parts_at_once)\
                                 .flatMap(lambda (x,y): zip(adtr.read_in_part(x, agg_data_shape, agg_mat_coords),
                                                            adtr.read_in_part(y,addl_agg_data_shape,addl_agg_mat_coords)))\
                                 .repartition(200)\
                                 .map(lambda ((idx1,x),(idx2,y)): (idx1,pl._merge_rdds_helper(x,y,addl_agg_to_agg_idx)))\
                                 .map(lambda (x,y): avg_impute((x,y,None,None), second_order_params))\
                                 .persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
#        imputed_part.count()

        #write imputation to bedgraph parts
        sorted_w_idx = imputed_part.mapPartitionsWithIndex(lambda x,y: pl._construct_bdg_parts(x, y, bdg_path, ct_list, assay_list, None, None, None, None, None, winsize=25, sinh=False, coords=requested_coords, tmpdir=tmpdir, enforce_nonegs=False)).count()
        imputed_part_tmp.unpersist()
        del(imputed_part_tmp)
#        imputed_part.unpersist()
#        del(imputed_part)

        #compile current round of bedgraph parts
        bdg_coord_glob = os.path.join(tmpdir, 'bdg_coords.*.txt')
        sc.parallelize([bdg_coord_glob], numSlices=1).foreach(pl._combine_bdg_coords)

#        bdg_path2 = os.path.join(tmpdir, '{0!s}_{1!s}/{0!s}_{1!s}.*.{2!s}.txt')

        sc.parallelize(requested_ctassay, numSlices=len(requested_ctassay)).mapPartitions(lambda x: pl._compile_bdg_batch_and_upload(x, out_bucket, out_root, bdg_path, tmpdir=tmpdir)).count()

        #remove the bdg_coords.txt file
        sc.parallelize([os.path.join(tmpdir, 'bdg_coords.txt')], numSlices=1).foreach(lambda x: os.remove(x))

        sc.stop()

    #start SparkContext
    sc = SparkContext(appName='impute_data_gen_bigwig',
                      pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                               pl.__file__.replace('.pyc', '.py'),
                               adtr.__file__.replace('.pyc', '.py')])
    pl.sc = sc
    adtr.pl = pl

    print('Generating bigwigs')
    bdg_url_glob = 's3://{!s}/{!s}'.format(out_bucket, os.path.join(out_root, '{0!s}/{1!s}/{0!s}_{1!s}.*.{2!s}.txt.gz'))
    out_bw_url = 's3://{!s}/{!s}'.format(out_bucket, os.path.join(out_root, '{0!s}_{1!s}.{2!s}.bw'))
    track_lines = sc.parallelize(requested_ctassay, numSlices=len(requested_ctassay)).map(lambda x: pl._compiled_bdg_batches_to_bigwig(x, bdg_url_glob, out_bw_url, make_public=args.make_public, tmpdir=tmpdir)).collect()

    out_url = 's3://{!s}/{!s}'.format(out_bucket, os.path.join(out_root, 'track_lines.txt'))
    with smart_open.smart_open(out_url, 'w') as out:
        out.write('\n'.join(sorted(track_lines)))

    sc.stop()

parser = argparse.ArgumentParser()
parser.add_argument('--model_url_glob', help='Glob describing S3 URI for model training results (use the hyperparameters.pickle file). This can describe a directory for a single PREDICTD run (e.g. the output of train_model.py) or multiple PREDICTD runs (e.g. the output of one or more instances of train_bags.py).')
parser.add_argument('--model_data_idx', help='S3 URI for the data_idx.pickle file from the model training. If multiple models are to be averaged, the data_idx data structure must be the same for all models.')

parser.add_argument('--agg_data_idx', help='S3 URI for the data_idx.pickle file from the assemble_new_ct_datasets.py and upload_genome_file_in_parts.py output.')
parser.add_argument('--agg_coord_order', help='The S3 URI for the *coord_order.pickle file generated by assemble_new_ct_datasets.py and upload_genome_file_in_parts.py.')
parser.add_argument('--agg_parts', help='The shared portion of the S3 URI for the assembled data parts produced by upload_genome_file_in_parts.py.')

parser.add_argument('--addl_agg_data_idx', help='If this model was trained on both the Roadmap Consolidated data tensor and a user-specified tensor with new data, provide the S3 URI of the data_idx.pickle file generated by assemble_new_ct_datasets.py and upload_genome_file_in_parts.py here.')
parser.add_argument('--addl_agg_coord_order', help='If this model was trained on both the Roadmap Consolidated data tensor and a user-specified tensor with new data, provide the S3 URI of the *coord_order.pickle file generated by assemble_new_ct_datasets.py and upload_genome_file_in_parts.py here.')

parser.add_argument('--cts_to_impute', help='Comma-delimited list of cell type names for which imputed data files should be generated. If no cell type names are specified, imputed data will be output for all cell types.')
parser.add_argument('--assays_to_impute', help='Comma-delimited list of assay names for which imputed data files should be generated. If no assay names are specified, imputed data will be output for all assays.')

parser.add_argument('--out_root_url', help='S3 URI root for all imputed data to share.')
parser.add_argument('--tmpdir', default='/data/tmp', help='Path to the local temporary directory to use for assembling the imputed data files. It must have enough storage capacity to store all of the requested imputed data. [default: %(default)s]')
parser.add_argument('--rdd_sort', action='store_true', default=False, help='In order to make a bigWig from a bedGraph, the bedGraph must be sorted by chromosomal coordinate. This sorting can either take place in the RDD during the calculation of imputed values or on the command line just before the script calls bedGraphToBigWig. If this option is selected, sorting will occur in memory in the RDD before the bedGraph is written. It is recommended to use this option if you are generating a large number of tracks (e.g. greater than the number of cores on the worker node of the cluster) or if there is an I/O bottleneck.')
parser.add_argument('--make_public', action='store_true', default=False, help='Set this option to make the bigWig files public in the output S3 bucket. This will facilitate their display on the UCSC Genome Browser.')
parser.add_argument('--coords_bed', help='A bed file containing regions of the genome to which the imputed data should be restricted (e.g. the ENCODE Pilot Regions). If this option is not set, then imputation will proceed for all records in the --agg_parts data source supplied.')
parser.add_argument('--parts_at_once', type=int, default=3, help='The number of --agg_parts files that should be processed per batch. This is a function of how many experiments you are imputing and how much memory and disk space is available on your system. [default: %(default)s, good for whole genome imputation (no --coords_bed specified) on an r3.8xlarge worker with --tmpdir set to /mnt2]')
parser.add_argument('--arcsinh', action='store_true', default=False, help='Whether or not to transform the data values from --agg_parts with the inverse hyperbolic sine transform. Note that this is usually not necessary at this stage because the transformation can be performed when generating the --agg_parts data.')
parser.add_argument('--random_seed', type=int, default=1, help='Seed value for the random number generator used for picking models to use for imputing missing data. Observed data is always imputed from models trained with an appropriate test set. [default: %(default)]')
parser.add_argument('--ignore_test_sets', action='store_true', default=False, help='Ignore the test/training split and impute a value for every requested experiment from every model. Note that this generally is not recommended, as the imputed tracks that correspond to experiments in the training set will not be valid predictions; however, it can be useful for comparison or if one just wants to see what all of the tracks look like.')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
