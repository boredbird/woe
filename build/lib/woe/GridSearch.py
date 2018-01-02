# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.svm import l1_min_c
from woe.eval import  compute_ks

"""
Search for optimal hyper parametric C in LogisticRegression
"""
def grid_search_lr_c(X_train,y_train,cs,df_coef_path=False
                     ,pic_coefpath_title='Logistic Regression Path',pic_coefpath=False
                     ,pic_performance_title='Logistic Regression Performance',pic_performance=False):
    """
    grid search optimal hyper parameters c with the best ks performance
    :param X_train: features dataframe
    :param y_train: target
    :param cs: list of regularization parameter c
    :param df_coef_path: the file path for logistic regression coefficient dataframe
    :param pic_coefpath_title: the pic title for coefficient path picture
    :param pic_coefpath: the file path for coefficient path picture
    :param pic_performance_title: the pic title for ks performance picture
    :param pic_performance: the file path for ks performance picture
    :return: a tuple of c and ks value with the best ks performance
    """
    # init a LogisticRegression model
    clf_l1_LR = LogisticRegression(C=0.1, penalty='l1', tol=0.01,class_weight='balanced')
    # cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 9,200)

    print("Computing regularization path ...")
    start = datetime.now()
    print start
    coefs_ = []
    ks = []
    for c in cs:
        clf_l1_LR.set_params(C=c)
        clf_l1_LR.fit(X_train, y_train)
        coefs_.append(clf_l1_LR.coef_.ravel().copy())

        proba = clf_l1_LR.predict_proba(X_train)[:,1]
        ks.append(compute_ks(proba,y_train))

    end = datetime.now()
    print end
    print("This took ", end - start)
    coef_cv_df = pd.DataFrame(coefs_,columns=X_train.columns)
    coef_cv_df['ks'] = ks
    coef_cv_df['c'] = cs

    if df_coef_path:
        file_name = df_coef_path if isinstance(df_coef_path, str) else None
        coef_cv_df.to_csv(file_name)

    coefs_ = np.array(coefs_)

    fig1 = plt.figure('fig1')
    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title(pic_coefpath_title)
    plt.axis('tight')
    if pic_coefpath:
        file_name = pic_coefpath if isinstance(pic_coefpath, str) else None
        plt.savefig(file_name)
    else:
        plt.show()

    fig2 = plt.figure('fig2')
    plt.plot(np.log10(cs), ks)
    plt.xlabel('log(C)')
    plt.ylabel('ks score')
    plt.title(pic_performance_title)
    plt.axis('tight')
    if pic_performance:
        file_name = pic_performance if isinstance(pic_performance, str) else None
        plt.savefig(file_name)
    else:
        plt.show()

    flag = coefs_<0
    idx = np.array(ks)[flag.sum(axis=1) == 0].argmax()

    return (cs[idx],ks[idx])