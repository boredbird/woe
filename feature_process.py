# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import numpy as np
from woe.config import  *
from collections import namedtuple


class node:
    '''树的节点的类
    '''
    def __init__(self,var_name=None,iv=0,split_point=None,right=None,left=None):
        self.var_name = var_name  # 用于切分数据集的属性的列索引值
        self.iv = iv  # 设置叶节点的iv值
        self.split_point = split_point  # 存储划分的值
        self.right = right  # 右子树
        self.left = left  # 左子树


#定义类及属性，用于结构化输出结果
class InfoValue(object):

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

#分类变量
class DisInfoValue(object):

    def __init__(self):
        self.var_name = None
        self.origin_value = []
        self.woe_before = []


class ModelTrain(object):

    def __init__(self):

        self.model = None
        self.feature_name_coef_map = None

        self.infovalue = None
        self.data_summary = None
        self.feature_detail = None

        self.model_summary = None
        self.feature_stability = None
        self.feature_summary = None
        self.model_stability = None
        self.segment_metrics = None
        self.gini_score = None
        self.ks_score = None

def change_feature_dtype(df,variable_type):
    """
    change feature data type by the variable_type specified in the config_af.py file
    :param df:
    :return:
    """
    if len(df.columns) == variable_type.shape[0]:
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
    else:
        print len(df.columns)
        print variable_type.shape[0]
        raise ValueError("the colums num of dataset_train and varibale_type is not equal")

    return

def check_point(df,var,split,min_sample):
    """
    检查分割点是否会造成有些分组样本量过小;
    如果存在分组样本量低于总样本量的5%，则与相邻分组合并直至超过5%为止;
    仅适用于连续值
    :param df:
    :param var:
    :param split:
    :return:
    """
    new_split = []
    if split is not None and len(split)>0:
        new_split.append(split[0])
        #尝试第一个分割点划分的左区间，不行就去掉
        pdf = df[df[var] <= split[0]]
        # print len(pdf)
        # print len(set(pdf['target']))
        if (len(pdf) < min_sample) or (len(set(pdf['target']))<=1):
            new_split.pop()
        for i in range(0,len(split)-1):
            pdf = df[(df[var] > split[i]) & (df[var] <= split[i+1])]
            if (len(pdf) < min_sample) or (len(set(pdf['target']))<=1):
                continue
            else:
                new_split.append(split[i+1])

        #剩余样本太少，则去掉最后一个分割掉
        if (len(df[df[var] > split[len(split)-1]])< min_sample) & (len(new_split)>1):
            new_split.pop()
        #剩余样本只有正样本或负样本，则去掉最后一个分割掉
        if len(set(df[df[var] > split[len(split)-1]]['target']))<=1 and len(new_split)>1:
            new_split.pop()

        #split只有一个取值，且没有比这个值更小的值，例如dd6_pos:-1
        if new_split == []:
            new_split = split
    else:
        pass

    return new_split


def calulate_iv(df,var,global_bt,global_gt):
    #calculate the iv and woe value without split
    a = df.loc[:,[var,'target']]
    b = a.groupby(['target']).count()
    bt = global_bt
    gt = global_gt
    bt_sub = 0
    gt_sub = 0

    groupdetail = namedtuple('groupdetail', ['woei','ivi','sub_total_num_percentage','positive_sample_num', 'negative_sample_num', 'positive_rate_in_sub_total','negative_rate_in_sub_total'])

    # print b
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
    #calculate woe,iv
    #br aka Bag Ratio,Bi/Bt;gr aka Good Ratio,Gi/Gt;
    #l* or *l named left dataset via the split;r* or *l named right dataset via the split;
    # bt = l1_cnt + r1_cnt
    # gt = l0_cnt + r0_cnt
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


def binning_data_split(df,var,global_bt,global_gt,min_sample,iv=0):
    """
    Specify the data split level and return the split value list
    :return:
    """
    sign = 1 #是否继续分割的标识符

    iv_var = InfoValue()

    #计算当前节点的iv（未分割时）
    gd = calulate_iv(df, var,global_bt,global_gt)

    woei, ivi = gd.woei,gd.ivi
    ivi = ivi.values[0]

    # print set(df[var])

    if len(set(df[var])) <=8:
        split = list(set(df[var]))
        split.sort()
        #分割点检查与处理
        split = check_point(df, var, split, min_sample)
        split.sort()
        iv_var.split_list = split

        sign = 0 #停止分割
        # print 'add new split point from line 224: ',split
        return node(split_point=split,iv=ivi)

    percent_value = list(set(np.percentile(df[var], range(100))))
    percent_value.sort()

    if len(percent_value) <=2:
        iv_var.split_list = list(set(percent_value)).sort()
        sign = 0  # 停止分割
        # print 'add new split point from line 233: ', percent_value
        return node(split_point=percent_value,iv=ivi)


    #init bestSplit_iv with zero
    #哨兵，尝试对当前节点分割
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
        # 只有正样本或负样本，则跳过
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

    # print '当前层级划分完毕！'

    #如果划分之后的iv大于当前未分割前的iv值，则是有效的分割，则递归
    # specified step learning rate 0.01
    if bestSplit_iv > ivi*1.01 and len(bestSplit_dataset_r) > min_sample and len(bestSplit_dataset_l) > min_sample:

        presplit_right = node()
        presplit_left = node()

        #判断右节点是否满足分割前提条件(貌似肯定满足不用判断)
        if len(bestSplit_dataset_r) < min_sample or len(set(bestSplit_dataset_r['target'])) == 1:
            presplit_right.iv = bestSplit_ivr
            sign = 0  # 停止分割
            # print 'presplit_right.iv: ', presplit_right.iv
            right = presplit_right
            # return presplit_right
        else:
            # print '进入节点 右！'
            right = binning_data_split(bestSplit_dataset_r,var,global_bt,global_gt,min_sample)

        # 判断左节点是否满足分割前提条件(貌似肯定满足不用判断)
        if len(bestSplit_dataset_l) < min_sample or len(set(bestSplit_dataset_l['target'])) == 1:
            presplit_left.iv = bestSplit_ivl
            sign = 0  # 停止分割
            # print 'presplit_left.iv: ', presplit_left.iv
            left = presplit_left
            # return presplit_left
        else:
            # print '进入节点 左！'
            left = binning_data_split(bestSplit_dataset_l,var,global_bt,global_gt,min_sample)

        # print 'add new split point from line 316: ', bestSplit_point

        return node(var_name=var,split_point=bestSplit_point,iv=ivi,left=left,right=right)

    else:
        return node(var_name=var,iv=ivi) #返回当前节点最为最终叶节点


# 生成器，输出分割点
def search(tree,split_list):
    # print tree.split_point
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
    #just for continuous variable
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
    return civ #结构化输出


def woe_trans(dvar,civ):
    # replace the var value with the given woe value
    print 'WoE Transformation:','{0: >40}'.format(civ.var_name)
    if not civ.is_discrete:
        var = dvar[:]
        if len(civ.woe_list)>1:
            split_list = []
            split_list.append(float("-inf"))
            split_list.extend([i for i in civ.split_list])
            split_list.append(float("inf"))

            for i in range(len(split_list)-2):
                var[(var > split_list[i]) & (var <= split_list[i+1])] = civ.woe_list[i]
        else:
            var[:] = civ.woe_list[0]
    else:
        var = dvar[:]
        split_map = {}
        for i in range(len(civ.split_list)):
            for j in range(len(civ.split_list[i])):
                split_map[civ.split_list[i][j]] = civ.woe_list[i]

        var = var.map(split_map)

    return var

def proc_woe_discrete(df,var,global_bt,global_gt,min_sample):
    # 分类变量处理
    print var
    div = DisInfoValue()
    div.var_name = var
    rdict = {}
    cpvar = df[var]

    for var_value in set(df[var]):
        # 此处用==判别，需首先做nan值填充处理
        df_temp = df[df[var] == var_value]
        gd = calulate_iv(df_temp,var,global_bt,global_gt)
        woei, ivi = gd.woei,gd.ivi
        print ivi
        div.origin_value.append(var_value)
        div.woe_before.append(woei)

        # rdict[var_value] = woei
        print var_value
        rdict[var_value] = woei.values[0]

    cpvar = cpvar.map(rdict)
    df[var] = cpvar

    iv_tree = binning_data_split(df,var,global_bt,global_gt,min_sample, iv=0)

    #遍历树，取出分割点
    split_list = []
    search(iv_tree, split_list)
    split_list = list(set([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # 分割点检查与处理
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()

    civ = format_iv_split(df, var, split_list,global_bt,global_gt)
    civ.is_discrete = 1

    var_value_list = list(set(df[var]))
    discrete_group_list = []

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

    # civ.split_list = split_list_temp[0:len(split_list_temp)-1].remove([])
    # civ.split_list = split_list_temp[0:len(split_list_temp)-1]
    civ.split_list = a

    return civ


def proc_woe_continuous(df,var,global_bt,global_gt,min_sample):
    # 连续变量处理
    print '-------- process ',var,'--------'
    iv_tree = binning_data_split(df, var,global_bt,global_gt,min_sample,iv=0)

    #遍历树，取出分割点
    split_list = []
    search(iv_tree, split_list)
    split_list = list(set([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # 分割点检查与处理
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()

    civ = format_iv_split(df, var,split_list,global_bt,global_gt)

    # print '-----woe trans '+var+'------'
    #woe值替换
    # df[var] = woe_trans(df[var],civ)

    return civ
