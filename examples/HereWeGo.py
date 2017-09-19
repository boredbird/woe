# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import os
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import woe.eval as eval

config_path = os.getcwd()+'\\woe\\examples\\config.csv'
data_path = os.getcwd()+'\\woe\\examples\\UCI_Credit_Card.csv'
cfg = config.config()
cfg.load_file(config_path,data_path)

for var in cfg.bin_var_list:
    # fill null
    cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0

# change feature dtypes
fp.change_feature_dtype(cfg.dataset_train, cfg.variable_type)

rst = []

# process woe transformation of continuous variables
for var in cfg.bin_var_list:
    rst.append(fp.proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))

# process woe transformation of discrete variables
for var in cfg.discrete_var_list:
    # fill null
    cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
    rst.append(fp.proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))

feature_detail = eval.eval_feature_detail(rst,'output_feature_detail.csv')