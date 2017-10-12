# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import numpy as np
from woe.config import  *
from collections import namedtuple
import copy

class node:
    '''Tree Node Class
    '''
    def __init__(self,var_name=None,iv=0,split_point=None,right=None,left=None):
        self.var_name = var_name  # The column index value of the attributes that are used to split data sets
        self.iv = iv  # The info value of the node
        self.split_point = split_point  # Store split points list
        self.right = right  # Right sub tree
        self.left = left  # Left sub tree


class InfoValue(object):
    '''
    InfoValue Class
    '''
    def __init__(self):
        self.var_name = []
        self.split_list = []
        self.iv = 0
        self.woe_list = []
        self.iv_list = []
        self.is_discrete = 0
        self.sub_total_sample_num = []
        self.positive_sample_num = []
        self.negative_sample_num = []
        self.sub_total_num_percentage = []
        self.positive_rate_in_sub_total = []
        self.negative_rate_in_sub_total = []

    def init(self,civ):
        self.var_name = civ.var_name
        self.split_list = civ.split_list
        self.iv = civ.iv
        self.woe_list = civ.woe_list
        self.iv_list = civ.iv_list
        self.is_discrete = civ.is_discrete
        self.sub_total_sample_num = civ.sub_total_sample_num
        self.positive_sample_num = civ.positive_sample_num
        self.negative_sample_num = civ.negative_sample_num
        self.sub_total_num_percentage = civ.sub_total_num_percentage
        self.positive_rate_in_sub_total = civ.positive_rate_in_sub_total
        self.negative_rate_in_sub_total = civ.negative_rate_in_sub_total


class DisInfoValue(object):
    '''
    A Class for the storage of discrete variables transformation information
    '''
    def __init__(self):
        self.var_name = None
        self.origin_value = []
        self.woe_before = []


def change_feature_dtype(df,variable_type):
    '''
    change feature data type by the variable_type DataFrame
    :param df: dataset DataFrame
    :param variable_type: the DataFrame about variables dtypes
    :return: None
    '''
    s = 'Changing Feature Dtypes'
    print s.center(60,'-')
    for vname in df.columns:
        try:
            df[vname] = df[vname].astype(variable_type.loc[vname,'v_type'])
            print vname,' '*(40-len(vname)),'{0: >10}'.format(variable_type.loc[vname,'v_type'])
        except Exception:
            print '[error]',vname

    s = 'Variable Dtypes Have Been Specified'
    print s.center(60,'-')

    return

def check_point(df,var,split,min_sample):
    """
    Check whether the segmentation points cause some packet samples to be too small;
    If there is a packet sample size of less than 5% of the total sample size,
    then merge with the adjacent packet until more than 5%;
    Applies only to continuous values
    :param df: Dataset DataFrame
    :param var: Variables list
    :param split: Split points list
    :param min_sample: Minimum packet sample size
    :return: The split points list checked out
    """
    new_split = []
    if split is not None and len(split)>0:
        new_split.append(split[0])
        # Try the left section of the first split point partition;
        # If not meet the conditions then the split point will be removed
        pdf = df[df[var] <= split[0]]
        if (len(pdf) < min_sample) or (len(np.unique(pdf['target']))<=1):
            new_split.pop()
        for i in range(0,len(split)-1):
            pdf = df[(df[var] > split[i]) & (df[var] <= split[i+1])]
            if (len(pdf) < min_sample) or (len(np.unique(pdf['target']))<=1):
                continue
            else:
                new_split.append(split[i+1])

        #If the remaining sample is too small then remove the last one
        if (len(df[df[var] > split[len(split)-1]])< min_sample) & (len(new_split)>1):
            new_split.pop()
        #If the remaining samples have only a positive or negative target then remove the last one
        if len(np.unique(df[df[var] > split[len(split)-1]]['target']))<=1 and len(new_split)>1:
            new_split.pop()

        #If the split list has only one value, and no smaller than this value
        if new_split == []:
            new_split = split
    else:
        pass

    return new_split


def calulate_iv(df,var,global_bt,global_gt):
    '''
    calculate the iv and woe value without split
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :return:
    '''
    a = df.loc[:,[var,'target']]
    b = a.groupby(['target']).count()
    bt = global_bt
    gt = global_gt
    bt_sub = 0
    gt_sub = 0

    groupdetail = namedtuple('groupdetail', ['woei','ivi','sub_total_num_percentage','positive_sample_num', 'negative_sample_num', 'positive_rate_in_sub_total','negative_rate_in_sub_total'])

    try:
        bri = (b.ix[1,:]+0.0001) * 1.0 / bt
        bt_sub = b.ix[1,:][0]
    except Exception:
        bri = (0 + 0.0001) * 1.0 / bt
        bt_sub = 0

    try:
        gri = (b.ix[0,:]+0.0001) * 1.0 / gt
        gt_sub = b.ix[0,:][0]
    except Exception:
        gri = (0 + 0.0001) * 1.0 / gt
        gt_sub = 0

    woei = np.log(bri / gri)
    ivi = (bri - gri) * woei

    gd = groupdetail(woei=woei
                     ,ivi=ivi
                     ,sub_total_num_percentage = len(a)*1.0/(bt+gt)
                     ,positive_sample_num=bt_sub
                     ,negative_sample_num=gt_sub
                     ,positive_rate_in_sub_total=bt_sub*1.0/len(a)
                     ,negative_rate_in_sub_total=gt_sub*1.0/len(a))

    return gd

def calculate_iv_split(df,var,split_point,global_bt,global_gt):
    """
    calculate the iv value with the specified split point
    note:
        the dataset should have variables:'target' which to be encapsulated if have time
    :return:
    """
    #split dataset
    dataset_r = df[df.loc[:,var] > split_point][[var,'target']]
    dataset_l = df[df.loc[:,var] <= split_point][[var,'target']]

    #calculate subset statistical frequency
    a = dataset_r.groupby(['target', ]).count().reset_index()
    a.rename(columns={var:'cnt'}, inplace = True)
    r0_cnt = sum(a[a['target']==0]['cnt'])
    r1_cnt = sum(a[a['target']==1]['cnt'])

    b = dataset_l.groupby(['target', ]).count().reset_index()
    b.rename(columns={var:'cnt'}, inplace = True)
    l0_cnt = sum(b[b['target']==0]['cnt'])
    l1_cnt = sum(b[b['target']==1]['cnt'])

    if r0_cnt == 0 or r1_cnt == 0 or l0_cnt == 0 or l1_cnt ==0:
        return 0,0,0,dataset_l,dataset_r,0,0

    bt = global_bt
    gt = global_gt
    lbr = l1_cnt*1.0/bt
    lgr = l0_cnt*1.0/gt
    woel = np.log(lbr/lgr)
    ivl = (lbr-lgr)*woel
    rbr = r1_cnt*1.0/bt
    rgr = r0_cnt*1.0/gt
    woer = np.log(rbr/rgr)
    ivr = (rbr-rgr)*woer
    iv = ivl+ivr

    return woel,woer,iv,dataset_l,dataset_r,ivl,ivr


def binning_data_split(df,var,global_bt,global_gt,min_sample,alpha=0.01):
    """
    Specify the data split level and return the split value list
    :return:
    """
    iv_var = InfoValue()
    # Calculates the IV of the current node before splitted
    gd = calulate_iv(df, var,global_bt,global_gt)

    woei, ivi = gd.woei,gd.ivi
    ivi = ivi.values[0]

    if len(np.unique(df[var])) <=8:
        split = list(np.unique(df[var]))
        split.sort()
        #Segmentation point checking and processing
        split = check_point(df, var, split, min_sample)
        split.sort()
        iv_var.split_list = split
        return node(split_point=split,iv=ivi)

    percent_value = list(np.unique(np.percentile(df[var], range(100))))
    percent_value.sort()

    if len(percent_value) <=2:
        iv_var.split_list = list(np.unique(percent_value)).sort()
        return node(split_point=percent_value,iv=ivi)

    # A sentry that attempts to split the current node
    # Init bestSplit_iv with zero
    bestSplit_iv = 0
    bestSplit_woel = []
    bestSplit_woer = []
    bestSplit_ivl = 0
    bestSplit_ivr = 0
    bestSplit_point = []
    bestSplit_dataset_l = pd.DataFrame()
    bestSplit_dataset_r = pd.DataFrame()

    #remove max value and min value in case dataset_r  or dataset_l will be null
    for point in percent_value[0:len(percent_value)-1]:
        # If there is only a sample or a negative sample, skip
        if len(set(df[df[var] > point]['target'])) == 1 or len(set(df[df[var] <= point]['target'])) == 1\
                or len(df[df[var] > point]) < min_sample or len(df[df[var] <= point]) < min_sample :
            continue

        woel, woer, iv, dataset_l, dataset_r, ivl, ivr = calculate_iv_split(df,var,point,global_bt,global_gt)

        if iv > bestSplit_iv:
            bestSplit_woel = woel
            bestSplit_woer = woer
            bestSplit_iv = iv
            bestSplit_point = point
            bestSplit_dataset_r = dataset_r
            bestSplit_dataset_l = dataset_l
            bestSplit_ivl = ivl
            bestSplit_ivr = ivr

    # If the IV after division is greater than the IV value before the current segmentation, the segmentation is valid and recursive
    # specified step learning rate 0.01
    if bestSplit_iv > ivi*(1+alpha) and len(bestSplit_dataset_r) > min_sample and len(bestSplit_dataset_l) > min_sample:
        presplit_right = node()
        presplit_left = node()

        # Determine whether the right node satisfies the segmentation prerequisite
        if len(bestSplit_dataset_r) < min_sample or len(set(bestSplit_dataset_r['target'])) == 1:
            presplit_right.iv = bestSplit_ivr
            right = presplit_right
        else:
            right = binning_data_split(bestSplit_dataset_r,var,global_bt,global_gt,min_sample,alpha=0.01)

        # Determine whether the left node satisfies the segmentation prerequisite
        if len(bestSplit_dataset_l) < min_sample or len(np.unique(bestSplit_dataset_l['target'])) == 1:
            presplit_left.iv = bestSplit_ivl
            left = presplit_left
        else:
            left = binning_data_split(bestSplit_dataset_l,var,global_bt,global_gt,min_sample,alpha=0.01)

        return node(var_name=var,split_point=bestSplit_point,iv=ivi,left=left,right=right)
    else:
        # Returns the current node as the final leaf node
        return node(var_name=var,iv=ivi)


def search(tree,split_list):
    '''
    search the tree node
    :param tree: a instance of Tree Node Class
    :return: split points list
    '''
    if isinstance(tree.split_point, list):
        split_list.extend(tree.split_point)
    else:
        split_list.append(tree.split_point)

    if tree.left is not None:
        search(tree.left,split_list)

    if tree.right is not None:
        search(tree.right,split_list)

    return split_list


def format_iv_split(df,var,split_list,global_bt,global_gt):
    '''
    Given the dataset DataFrame and split points list then return a InfoValue instance;
    Just for continuous variable
    :param df:
    :param var:
    :param split_list:
    :param global_bt:
    :param global_gt:
    :return:
    '''
    civ = InfoValue()
    civ.var_name = var
    civ.split_list = split_list
    dfcp = df[:]

    civ.sub_total_sample_num = []
    civ.positive_sample_num = []
    civ.negative_sample_num = []
    civ.sub_total_num_percentage = []
    civ.positive_rate_in_sub_total = []

    for i in range(0, len(split_list)):
        dfi = dfcp[dfcp[var] <= split_list[i]]
        dfcp = dfcp[dfcp[var] > split_list[i]]
        gd = calulate_iv(dfi, var,global_bt,global_gt)
        woei, ivi = gd.woei,gd.ivi
        civ.woe_list.extend(woei)
        civ.iv_list.extend(ivi)
        civ.sub_total_sample_num.append(len(dfi))
        civ.positive_sample_num.append(gd.positive_sample_num)
        civ.negative_sample_num.append(gd.negative_sample_num)
        civ.sub_total_num_percentage.append(gd.sub_total_num_percentage)
        civ.positive_rate_in_sub_total.append(gd.positive_rate_in_sub_total)
        civ.negative_rate_in_sub_total.append(gd.negative_rate_in_sub_total)

    if len(dfcp)>0:
        gd = calulate_iv(dfcp, var,global_bt,global_gt)
        woei, ivi = gd.woei,gd.ivi
        civ.woe_list.extend(woei)
        civ.iv_list.extend(ivi)
        civ.sub_total_sample_num.append(len(dfcp))
        civ.positive_sample_num.append(gd.positive_sample_num)
        civ.negative_sample_num.append(gd.negative_sample_num)
        civ.sub_total_num_percentage.append(gd.sub_total_num_percentage)
        civ.positive_rate_in_sub_total.append(gd.positive_rate_in_sub_total)
        civ.negative_rate_in_sub_total.append(gd.negative_rate_in_sub_total)

    civ.iv = sum(civ.iv_list)
    return civ


def woe_trans(dvar,civ):
    # replace the var value with the given woe value
    var = copy.deepcopy(dvar)
    if not civ.is_discrete:
        if len(civ.woe_list)>1:
            split_list = []
            split_list.append(float("-inf"))
            split_list.extend([i for i in civ.split_list])
            split_list.append(float("inf"))

            for i in range(len(civ.woe_list)):
                var[(dvar > split_list[i]) & (dvar <= split_list[i+1])] = civ.woe_list[i]
        else:
            var[:] = civ.woe_list[0]
    else:
        split_map = {}
        for i in range(len(civ.split_list)):
            for j in range(len(civ.split_list[i])):
                split_map[civ.split_list[i][j]] = civ.woe_list[i]

        var = var.map(split_map)

    return var

def proc_woe_discrete(df,var,global_bt,global_gt,min_sample,alpha=0.01):
    '''
    process woe transformation of discrete variables
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'process discrete variable:'+str(var)
    print s.center(60, '-')
    div = DisInfoValue()
    div.var_name = var
    rdict = {}
    cpvar = df[var]

    for var_value in np.unique(df[var]):
        # Here come with a '==',in case type error you must do Nan filling process firstly
        df_temp = df[df[var] == var_value]
        gd = calulate_iv(df_temp,var,global_bt,global_gt)
        woei, ivi = gd.woei,gd.ivi
        div.origin_value.append(var_value)
        div.woe_before.append(woei)
        rdict[var_value] = woei.values[0]

    cpvar = cpvar.map(rdict)
    df[var] = cpvar

    iv_tree = binning_data_split(df,var,global_bt,global_gt,min_sample,alpha)

    # Traversal tree, get the segmentation point
    split_list = []
    search(iv_tree, split_list)
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # Segmentation point checking and processing
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()

    civ = format_iv_split(df, var, split_list,global_bt,global_gt)
    civ.is_discrete = 1

    split_list_temp = []
    split_list_temp.append(float("-inf"))
    split_list_temp.extend([i for i in split_list])
    split_list_temp.append(float("inf"))

    a = []
    for i in range(len(split_list_temp) - 1):
        temp = []
        for j in range(len(div.origin_value)):
            if (div.woe_before[j]>split_list_temp[i]).values[0] & (div.woe_before[j]<=split_list_temp[i+1]).values[0]:
                temp.append(div.origin_value[j])

        if temp != [] :
            a.append(temp)

    civ.split_list = a

    return civ


def proc_woe_continuous(df,var,global_bt,global_gt,min_sample,alpha=0.01):
    '''
    process woe transformation of discrete variables
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'process continuous variable:'+str(var)
    print s.center(60, '-')
    iv_tree = binning_data_split(df, var,global_bt,global_gt,min_sample,alpha)

    # Traversal tree, get the segmentation point
    split_list = []
    search(iv_tree, split_list)
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # Segmentation point checking and processing
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()

    civ = format_iv_split(df, var,split_list,global_bt,global_gt)

    return civ
