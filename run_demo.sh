#! /usr/bin/env bash

out_url=$1
strip_prot=${out_url#s3://}
out_bucket=${strip_prot%%/*}
out_key=${strip_prot#*/}

DATA_URL=s3://predictd/Observed/training_data.encodePilots_and_nchars.rdd.pickle

spark-submit ./train_model.py --data_url=$DATA_URL --run_bucket=$out_bucket --out_root=$out_key --fold_idx=0 --valid_fold_idx=0 --factor_init_seed=123 -f 100 -l 0.00450541331031 --beta1=0.9 --beta2=0.999 --epsilon=1e-8 --batch_size=5000 --win_per_slice=1000 --rc=4.79186469703 --ra=8.7565107627e-27 --ri=8.7565107627e-27 --ri2=0.412178570144 --rbc=0 --rba=0 --rbi=0 --stop_winsize=15 --stop_winspacing=5 --stop_win2shift=1e-05 --stop_pval=0.05 --min_iters=50 --iters_per_mse=3 --lrate_search_num=3 --folds_fname=s3://predictd/reference_data/folds.5.8.pickle --no_bw_sinh --training_fraction=0.01 --burn_in_epochs=0.5
