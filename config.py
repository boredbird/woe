# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import pandas as pd

class config:

    def __init__(self):
        self.config = None
        self.dataset_train = None
        self.variable_type = None
        self.bin_var_list = None
        self.discrete_var_list = None
        self.candidate_var_list = None
        self.dataset_len = None
        self.min_sample = None
        self.global_bt = None
        self.global_gt = None

    def load_file(self,config_path,data_path):
        self.config = pd.read_csv(config_path)
        # specify variable dtypes
        self.variable_type = self.config[['var_name', 'var_dtype']]
        self.variable_type = self.variable_type.rename(columns={'var_name': 'v_name', 'var_dtype': 'v_type'})
        self.variable_type = self.variable_type.set_index(['v_name'])

        # load dataset train
        self.dataset_train = pd.read_csv(data_path)
        self.dataset_train.columns = [col.split('.')[-1] for col in self.dataset_train.columns]

        # specify the list of continuous variable to be splitted into bin
        self.bin_var_list = self.config[self.config['is_tobe_bin'] == 1]['var_name']
        # specify the list of discrete variable to be merged into supper classes
        self.discrete_var_list = self.config[(self.config['is_candidate'] == 1) & (self.config['var_dtype'] == 'object')]['var_name']

        # specify the list of model input variable
        self.candidate_var_list = self.config[self.config['is_candidate'] == 1]['var_name']

        # specify some other global variables about the training dataset
        self.dataset_len = len(self.dataset_train)
        self.min_sample = int(self.dataset_len * 0.05)
        self.global_bt = sum(self.dataset_train['target'])
        self.global_gt = len(self.dataset_train) - sum(self.dataset_train['target'])








