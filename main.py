# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import woe.eval as eval

config_path = 'E://Code//ScoreCard//config//config_af_addr_call.csv'
data_path = 'E://ScoreCard//rawdata//af_policy_addr_call_f.csv'
cfg = config.config()
cfg.load_file(config_path,data_path)

for var in cfg.bin_var_list:
    # 空值处理
    cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0

# 改变数据类型
fp.change_feature_dtype(cfg.dataset_train, cfg.variable_type)

rst = []  # InfoValue类实例列表，用于存储每个变量的分割点、woe、iv等信息

# 连续变量处理
for var in cfg.bin_var_list:
    rst.append(fp.proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample))

# 分类变量处理
for var in cfg.discrete_var_list:
    # 空值处理
    cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
    rst.append(fp.proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample))

feature_detail = eval.eval_feature_detail(rst)


# ###########################加载测试集
dataset_validation = pd.read_csv(data_path)
dataset_validation.columns = [col.split('.')[-1] for col in dataset_validation.columns]

for var in cfg.discrete_var_list:
    # 空值处理
    dataset_validation.loc[dataset_validation[var].isnull(), (var)] = 'missing'

for r in rst:
    dataset_validation[r.var_name] = fp.woe_trans(dataset_validation[r.var_name],r)

##############################################
X = dataset_validation[cfg.candidate_var_list]
y = dataset_validation['target']

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)#axis:0:列，1：行
imp.fit(X)
X = imp.transform(X)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

X = cfg.dataset_train[cfg.candidate_var_list]
y = cfg.dataset_train['target']
model.fit(X, y)
model.score(X, y)

proba = model.predict_proba(X)[:, 1]
eval.compute_ks_gini(dataset_validation['target'], proba, segment_cnt = 100, plot = False)

