import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
# np.random.seed(6)
np.random.seed(6)

def load_data_det_8(train_data, test_data):
    #########腾哥的特征选择
    train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    train_X = train_X.values
    test_X = test_X.values

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

    train_Y = train_data.iloc[:, 0]
    test_Y = test_data.iloc[:, 0]

    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1, 1)
    test_Y = test_Y.reshape(-1, 1)

    return train_X, train_Y, test_X, test_Y

def feature_selection(train_data, test_data):
    #########腾哥的特征选择
    # train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    train_X = train_data[['SF-WAT', 'SA-CFM', 'RA-CFM', 'SA-TEMP', 'MA-TEMP',  'OA-TEMP', 'HWC-EWT','E_ccoil']]
    test_X = test_data[['SF-WAT', 'SA-CFM', 'RA-CFM', 'SA-TEMP', 'MA-TEMP',   'OA-TEMP', 'HWC-EWT','E_ccoil']]
    train_X = train_X.values
    test_X = test_X.values

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

    train_Y = train_data.iloc[:, 0]
    test_Y = test_data.iloc[:, 0]

    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1, 1)
    test_Y = test_Y.reshape(-1, 1)

    return train_X, train_Y, test_X, test_Y

#############################故障等级一###############################################################################################

lev1_lab0 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab0.csv')  # D:\zhuo mian wen jian\maching\brother\tengge\新想法1\数据集孙
lev1_lab1 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab1.csv')
lev1_lab2 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab2.csv')
lev1_lab3 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab3.csv')
lev1_lab4 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab4.csv')
lev1_lab5 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab5.csv')
lev1_lab6 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab6.csv')
lev1_lab7 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab7.csv')

# 丢弃包含NAN的数据
lev1_lab0 = lev1_lab0.dropna()
lev1_lab1 = lev1_lab1.dropna()
lev1_lab2 = lev1_lab2.dropna()
lev1_lab3 = lev1_lab3.dropna()
lev1_lab4 = lev1_lab4.dropna()
lev1_lab5 = lev1_lab5.dropna()
lev1_lab6 = lev1_lab6.dropna()
lev1_lab7 = lev1_lab7.dropna()

lev1_lab0_1 = lev1_lab0.iloc[0:5000, :]
lev1_lab1_1 = lev1_lab1.iloc[0:5000, :]
lev1_lab2_1 = lev1_lab2.iloc[0:5000, :]
lev1_lab3_1 = lev1_lab3.iloc[0:5000, :]
lev1_lab4_1 = lev1_lab4.iloc[0:5000, :]
lev1_lab5_1 = lev1_lab5.iloc[0:5000, :]
lev1_lab6_1 = lev1_lab6.iloc[0:5000, :]
lev1_lab7_1 = lev1_lab7.iloc[0:5000, :]



lev1_lab0_train_1cv = lev1_lab0_1.iloc[1000:5000, :]
lev1_lab0_test_1cv = lev1_lab0_1.iloc[0:1000, :]
lev1_lab1_train_1cv = lev1_lab1_1.iloc[1000:5000, :]
lev1_lab1_test_1cv = lev1_lab1_1.iloc[0:1000, :]
lev1_lab2_train_1cv = lev1_lab2_1.iloc[1000:5000, :]
lev1_lab2_test_1cv = lev1_lab2_1.iloc[0:1000, :]
lev1_lab3_train_1cv = lev1_lab3_1.iloc[1000:5000, :]
lev1_lab3_test_1cv = lev1_lab3_1.iloc[0:1000, :]
lev1_lab4_train_1cv = lev1_lab4_1.iloc[1000:5000, :]
lev1_lab4_test_1cv = lev1_lab4_1.iloc[0:1000, :]
lev1_lab5_train_1cv = lev1_lab5_1.iloc[1000:5000, :]
lev1_lab5_test_1cv = lev1_lab5_1.iloc[0:1000, :]
lev1_lab6_train_1cv = lev1_lab6_1.iloc[1000:5000, :]
lev1_lab6_test_1cv = lev1_lab6_1.iloc[0:1000, :]
lev1_lab7_train_1cv = lev1_lab7_1.iloc[1000:5000, :]
lev1_lab7_test_1cv = lev1_lab7_1.iloc[0:1000, :]

# 合并训练集测试集
lev1_train_1cv = pd.concat([lev1_lab0_train_1cv, lev1_lab1_train_1cv, lev1_lab2_train_1cv, lev1_lab3_train_1cv,
                        lev1_lab4_train_1cv, lev1_lab5_train_1cv, lev1_lab6_train_1cv, lev1_lab7_train_1cv], axis=0)
lev1_test_1cv = pd.concat([lev1_lab0_test_1cv, lev1_lab1_test_1cv, lev1_lab2_test_1cv, lev1_lab3_test_1cv,
                       lev1_lab4_test_1cv, lev1_lab5_test_1cv, lev1_lab6_test_1cv, lev1_lab7_test_1cv], axis=0)


lev1_1cv_train_X, lev1_1cv_train_Y,lev1_1cv_test_X, lev1_1cv_test_Y = load_data_det_8(lev1_train_1cv,lev1_test_1cv)






lev1_lab0_train_2cv_11 = lev1_lab0_1.iloc[0:1000, :]
lev1_lab0_train_2cv_12 = lev1_lab0_1.iloc[2000:5000, :]
lev1_lab0_test_2cv = lev1_lab0_1.iloc[1000:2000, :]

lev1_lab1_train_2cv_11 = lev1_lab1_1.iloc[0:1000, :]
lev1_lab1_train_2cv_12 = lev1_lab1_1.iloc[2000:5000, :]
lev1_lab1_test_2cv = lev1_lab1_1.iloc[1000:2000, :]

lev1_lab2_train_2cv_11 = lev1_lab2_1.iloc[0:1000, :]
lev1_lab2_train_2cv_12 = lev1_lab2_1.iloc[2000:5000, :]
lev1_lab2_test_2cv = lev1_lab2_1.iloc[1000:2000, :]

lev1_lab3_train_2cv_11 = lev1_lab3_1.iloc[0:1000, :]
lev1_lab3_train_2cv_12 = lev1_lab3_1.iloc[2000:5000, :]
lev1_lab3_test_2cv = lev1_lab3_1.iloc[1000:2000, :]

lev1_lab4_train_2cv_11 = lev1_lab4_1.iloc[0:1000, :]
lev1_lab4_train_2cv_12 = lev1_lab4_1.iloc[2000:5000, :]
lev1_lab4_test_2cv = lev1_lab4_1.iloc[1000:2000, :]

lev1_lab5_train_2cv_11 = lev1_lab5_1.iloc[0:1000, :]
lev1_lab5_train_2cv_12 = lev1_lab5_1.iloc[2000:5000, :]
lev1_lab5_test_2cv = lev1_lab5_1.iloc[1000:2000, :]


lev1_lab6_train_2cv_11 = lev1_lab6_1.iloc[0:1000, :]
lev1_lab6_train_2cv_12 = lev1_lab6_1.iloc[2000:5000, :]
lev1_lab6_test_2cv = lev1_lab6_1.iloc[1000:2000, :]

lev1_lab7_train_2cv_11 = lev1_lab7_1.iloc[0:1000, :]
lev1_lab7_train_2cv_12 = lev1_lab7_1.iloc[2000:5000, :]
lev1_lab7_test_2cv = lev1_lab7_1.iloc[1000:2000, :]


lev1_train_2cv = pd.concat([lev1_lab0_train_2cv_11, lev1_lab0_train_2cv_12,lev1_lab1_train_2cv_11, lev1_lab1_train_2cv_12,
                          lev1_lab2_train_2cv_11, lev1_lab2_train_2cv_12,lev1_lab3_train_2cv_11, lev1_lab3_train_2cv_12,
                          lev1_lab4_train_2cv_11, lev1_lab4_train_2cv_12,lev1_lab5_train_2cv_11, lev1_lab5_train_2cv_12,
                          lev1_lab6_train_2cv_11, lev1_lab6_train_2cv_12,lev1_lab7_train_2cv_11, lev1_lab7_train_2cv_12], axis=0)
lev1_test_2cv = pd.concat([lev1_lab0_test_2cv,lev1_lab1_test_2cv,lev1_lab2_test_2cv,lev1_lab3_test_2cv,lev1_lab4_test_2cv,lev1_lab5_test_2cv,
                         lev1_lab6_test_2cv,lev1_lab7_test_2cv], axis=0)


lev1_2cv_train_X, lev1_2cv_train_Y,lev1_2cv_test_X, lev1_2cv_test_Y = load_data_det_8(lev1_train_2cv,lev1_test_2cv)





lev1_lab0_train_3cv_11 = lev1_lab0_1.iloc[0:2000, :]
lev1_lab0_train_3cv_12 = lev1_lab0_1.iloc[3000:5000, :]
lev1_lab0_test_3cv = lev1_lab0_1.iloc[2000:3000, :]

lev1_lab1_train_3cv_11 = lev1_lab1_1.iloc[0:2000, :]
lev1_lab1_train_3cv_12 = lev1_lab1_1.iloc[3000:5000, :]
lev1_lab1_test_3cv = lev1_lab1_1.iloc[2000:3000, :]

lev1_lab2_train_3cv_11 = lev1_lab2_1.iloc[0:2000, :]
lev1_lab2_train_3cv_12 = lev1_lab2_1.iloc[3000:5000, :]
lev1_lab2_test_3cv = lev1_lab2_1.iloc[2000:3000, :]

lev1_lab3_train_3cv_11 = lev1_lab3_1.iloc[0:2000, :]
lev1_lab3_train_3cv_12 = lev1_lab3_1.iloc[3000:5000, :]
lev1_lab3_test_3cv = lev1_lab3_1.iloc[2000:3000, :]

lev1_lab4_train_3cv_11 = lev1_lab4_1.iloc[0:2000, :]
lev1_lab4_train_3cv_12 = lev1_lab4_1.iloc[3000:5000, :]
lev1_lab4_test_3cv = lev1_lab4_1.iloc[2000:3000, :]

lev1_lab5_train_3cv_11 = lev1_lab5_1.iloc[0:2000, :]
lev1_lab5_train_3cv_12 = lev1_lab5_1.iloc[3000:5000, :]
lev1_lab5_test_3cv = lev1_lab5_1.iloc[2000:3000, :]


lev1_lab6_train_3cv_11 = lev1_lab6_1.iloc[0:2000, :]
lev1_lab6_train_3cv_12 = lev1_lab6_1.iloc[3000:5000, :]
lev1_lab6_test_3cv = lev1_lab6_1.iloc[2000:3000, :]

lev1_lab7_train_3cv_11 = lev1_lab7_1.iloc[0:2000, :]
lev1_lab7_train_3cv_12 = lev1_lab7_1.iloc[3000:5000, :]
lev1_lab7_test_3cv = lev1_lab7_1.iloc[2000:3000, :]


lev1_train_3cv = pd.concat([lev1_lab0_train_3cv_11, lev1_lab0_train_3cv_12,lev1_lab1_train_3cv_11, lev1_lab1_train_3cv_12,
                          lev1_lab2_train_3cv_11, lev1_lab2_train_3cv_12,lev1_lab3_train_3cv_11, lev1_lab3_train_3cv_12,
                          lev1_lab4_train_3cv_11, lev1_lab4_train_3cv_12,lev1_lab5_train_3cv_11, lev1_lab5_train_3cv_12,
                          lev1_lab6_train_3cv_11, lev1_lab6_train_3cv_12,lev1_lab7_train_3cv_11, lev1_lab7_train_3cv_12], axis=0)
lev1_test_3cv = pd.concat([lev1_lab0_test_3cv,lev1_lab1_test_3cv,lev1_lab2_test_3cv,lev1_lab3_test_3cv,lev1_lab4_test_3cv,lev1_lab5_test_3cv,
                         lev1_lab6_test_3cv,lev1_lab7_test_3cv], axis=0)


lev1_3cv_train_X, lev1_3cv_train_Y,lev1_3cv_test_X, lev1_3cv_test_Y = load_data_det_8(lev1_train_3cv,lev1_test_3cv)



lev1_lab0_train_4cv_11 = lev1_lab0_1.iloc[0:3000, :]
lev1_lab0_train_4cv_12 = lev1_lab0_1.iloc[4000:5000, :]
lev1_lab0_test_4cv = lev1_lab0_1.iloc[3000:4000, :]

lev1_lab1_train_4cv_11 = lev1_lab1_1.iloc[0:3000, :]
lev1_lab1_train_4cv_12 = lev1_lab1_1.iloc[4000:5000, :]
lev1_lab1_test_4cv = lev1_lab1_1.iloc[3000:4000, :]

lev1_lab2_train_4cv_11 = lev1_lab2_1.iloc[0:3000, :]
lev1_lab2_train_4cv_12 = lev1_lab2_1.iloc[4000:5000, :]
lev1_lab2_test_4cv = lev1_lab2_1.iloc[3000:4000, :]

lev1_lab3_train_4cv_11 = lev1_lab3_1.iloc[0:3000, :]
lev1_lab3_train_4cv_12 = lev1_lab3_1.iloc[4000:5000, :]
lev1_lab3_test_4cv = lev1_lab3_1.iloc[3000:4000, :]

lev1_lab4_train_4cv_11 = lev1_lab4_1.iloc[0:3000, :]
lev1_lab4_train_4cv_12 = lev1_lab4_1.iloc[4000:5000, :]
lev1_lab4_test_4cv = lev1_lab4_1.iloc[3000:4000, :]

lev1_lab5_train_4cv_11 = lev1_lab5_1.iloc[0:3000, :]
lev1_lab5_train_4cv_12 = lev1_lab5_1.iloc[4000:5000, :]
lev1_lab5_test_4cv = lev1_lab5_1.iloc[3000:4000, :]


lev1_lab6_train_4cv_11 = lev1_lab6_1.iloc[0:3000, :]
lev1_lab6_train_4cv_12 = lev1_lab6_1.iloc[4000:5000, :]
lev1_lab6_test_4cv = lev1_lab6_1.iloc[3000:4000, :]

lev1_lab7_train_4cv_11 = lev1_lab7_1.iloc[0:3000, :]
lev1_lab7_train_4cv_12 = lev1_lab7_1.iloc[4000:5000, :]
lev1_lab7_test_4cv = lev1_lab7_1.iloc[3000:4000, :]


lev1_train_4cv = pd.concat([lev1_lab0_train_4cv_11, lev1_lab0_train_4cv_12,lev1_lab1_train_4cv_11, lev1_lab1_train_4cv_12,
                          lev1_lab2_train_4cv_11, lev1_lab2_train_4cv_12,lev1_lab3_train_4cv_11, lev1_lab3_train_4cv_12,
                          lev1_lab4_train_4cv_11, lev1_lab4_train_4cv_12,lev1_lab5_train_4cv_11, lev1_lab5_train_4cv_12,
                          lev1_lab6_train_4cv_11, lev1_lab6_train_4cv_12,lev1_lab7_train_4cv_11, lev1_lab7_train_4cv_12], axis=0)
lev1_test_4cv = pd.concat([lev1_lab0_test_4cv,lev1_lab1_test_4cv,lev1_lab2_test_4cv,lev1_lab3_test_4cv,lev1_lab4_test_4cv,lev1_lab5_test_4cv,
                         lev1_lab6_test_4cv,lev1_lab7_test_4cv], axis=0)


lev1_4cv_train_X, lev1_4cv_train_Y,lev1_4cv_test_X, lev1_4cv_test_Y = load_data_det_8(lev1_train_4cv,lev1_test_4cv)





# 将n0个数据划分为训练集、验证集、测试集(8:1:1)
lev1_lab0_train_5cv = lev1_lab0_1.iloc[0:4000, :]
lev1_lab0_test_5cv = lev1_lab0_1.iloc[4000:5000, :]
lev1_lab1_train_5cv = lev1_lab1_1.iloc[0:4000, :]
lev1_lab1_test_5cv = lev1_lab1_1.iloc[4000:5000, :]
lev1_lab2_train_5cv = lev1_lab2_1.iloc[0:4000, :]
lev1_lab2_test_5cv = lev1_lab2_1.iloc[4000:5000, :]
lev1_lab3_train_5cv = lev1_lab3_1.iloc[0:4000, :]
lev1_lab3_test_5cv = lev1_lab3_1.iloc[4000:5000, :]
lev1_lab4_train_5cv = lev1_lab4_1.iloc[0:4000, :]
lev1_lab4_test_5cv = lev1_lab4_1.iloc[4000:5000, :]
lev1_lab5_train_5cv = lev1_lab5_1.iloc[0:4000, :]
lev1_lab5_test_5cv = lev1_lab5_1.iloc[4000:5000, :]
lev1_lab6_train_5cv = lev1_lab6_1.iloc[0:4000, :]
lev1_lab6_test_5cv = lev1_lab6_1.iloc[4000:5000, :]
lev1_lab7_train_5cv = lev1_lab7_1.iloc[0:4000, :]
lev1_lab7_test_5cv = lev1_lab7_1.iloc[4000:5000, :]

# 合并训练集测试集
lev1_train_5cv = pd.concat([lev1_lab0_train_5cv, lev1_lab1_train_5cv, lev1_lab2_train_5cv, lev1_lab3_train_5cv,
                        lev1_lab4_train_5cv, lev1_lab5_train_5cv, lev1_lab6_train_5cv, lev1_lab7_train_5cv], axis=0)
lev1_test_5cv = pd.concat([lev1_lab0_test_5cv, lev1_lab1_test_5cv, lev1_lab2_test_5cv, lev1_lab3_test_5cv,
                       lev1_lab4_test_5cv, lev1_lab5_test_5cv, lev1_lab6_test_5cv, lev1_lab7_test_5cv], axis=0)


lev1_5cv_train_X, lev1_5cv_train_Y,lev1_5cv_test_X, lev1_5cv_test_Y = load_data_det_8(lev1_train_5cv,lev1_test_5cv)




lev2_lab0 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab0.csv')
lev2_lab1 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab1.csv')
lev2_lab2 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab2.csv')
lev2_lab3 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab3.csv')
lev2_lab4 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab4.csv')
lev2_lab5 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab5.csv')
lev2_lab6 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab6.csv')
lev2_lab7 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab7.csv')

lev2_lab0 = lev2_lab0.dropna()
lev2_lab1 = lev2_lab1.dropna()
lev2_lab2 = lev2_lab2.dropna()
lev2_lab3 = lev2_lab3.dropna()
lev2_lab4 = lev2_lab4.dropna()
lev2_lab5 = lev2_lab5.dropna()
lev2_lab6 = lev2_lab6.dropna()
lev2_lab7 = lev2_lab7.dropna()

lev2_lab0_1 = lev2_lab0.iloc[0:5000, :]
lev2_lab1_1 = lev2_lab1.iloc[0:5000, :]
lev2_lab2_1 = lev2_lab2.iloc[0:5000, :]
lev2_lab3_1 = lev2_lab3.iloc[0:5000, :]
lev2_lab4_1 = lev2_lab4.iloc[0:5000, :]
lev2_lab5_1 = lev2_lab5.iloc[0:5000, :]
lev2_lab6_1 = lev2_lab6.iloc[0:5000, :]
lev2_lab7_1 = lev2_lab7.iloc[0:5000, :]



lev2_lab0_train_1cv = lev2_lab0_1.iloc[1000:5000, :]
lev2_lab0_test_1cv = lev2_lab0_1.iloc[0:1000, :]
lev2_lab1_train_1cv = lev2_lab1_1.iloc[1000:5000, :]
lev2_lab1_test_1cv = lev2_lab1_1.iloc[0:1000, :]
lev2_lab2_train_1cv = lev2_lab2_1.iloc[1000:5000, :]
lev2_lab2_test_1cv = lev2_lab2_1.iloc[0:1000, :]
lev2_lab3_train_1cv = lev2_lab3_1.iloc[1000:5000, :]
lev2_lab3_test_1cv = lev2_lab3_1.iloc[0:1000, :]
lev2_lab4_train_1cv = lev2_lab4_1.iloc[1000:5000, :]
lev2_lab4_test_1cv = lev2_lab4_1.iloc[0:1000, :]
lev2_lab5_train_1cv = lev2_lab5_1.iloc[1000:5000, :]
lev2_lab5_test_1cv = lev2_lab5_1.iloc[0:1000, :]
lev2_lab6_train_1cv = lev2_lab6_1.iloc[1000:5000, :]
lev2_lab6_test_1cv = lev2_lab6_1.iloc[0:1000, :]
lev2_lab7_train_1cv = lev2_lab7_1.iloc[1000:5000, :]
lev2_lab7_test_1cv = lev2_lab7_1.iloc[0:1000, :]

# 合并训练集测试集
lev2_train_1cv = pd.concat([lev2_lab0_train_1cv, lev2_lab1_train_1cv, lev2_lab2_train_1cv, lev2_lab3_train_1cv,
                        lev2_lab4_train_1cv, lev2_lab5_train_1cv, lev2_lab6_train_1cv, lev2_lab7_train_1cv], axis=0)
lev2_test_1cv = pd.concat([lev2_lab0_test_1cv, lev2_lab1_test_1cv, lev2_lab2_test_1cv, lev2_lab3_test_1cv,
                       lev2_lab4_test_1cv, lev2_lab5_test_1cv, lev2_lab6_test_1cv, lev2_lab7_test_1cv], axis=0)


lev2_1cv_train_X, lev2_1cv_train_Y,lev2_1cv_test_X, lev2_1cv_test_Y = load_data_det_8(lev2_train_1cv,lev2_test_1cv)






lev2_lab0_train_2cv_11 = lev2_lab0_1.iloc[0:1000, :]
lev2_lab0_train_2cv_12 = lev2_lab0_1.iloc[2000:5000, :]
lev2_lab0_test_2cv = lev2_lab0_1.iloc[1000:2000, :]

lev2_lab1_train_2cv_11 = lev2_lab1_1.iloc[0:1000, :]
lev2_lab1_train_2cv_12 = lev2_lab1_1.iloc[2000:5000, :]
lev2_lab1_test_2cv = lev2_lab1_1.iloc[1000:2000, :]

lev2_lab2_train_2cv_11 = lev2_lab2_1.iloc[0:1000, :]
lev2_lab2_train_2cv_12 = lev2_lab2_1.iloc[2000:5000, :]
lev2_lab2_test_2cv = lev2_lab2_1.iloc[1000:2000, :]

lev2_lab3_train_2cv_11 = lev2_lab3_1.iloc[0:1000, :]
lev2_lab3_train_2cv_12 = lev2_lab3_1.iloc[2000:5000, :]
lev2_lab3_test_2cv = lev2_lab3_1.iloc[1000:2000, :]

lev2_lab4_train_2cv_11 = lev2_lab4_1.iloc[0:1000, :]
lev2_lab4_train_2cv_12 = lev2_lab4_1.iloc[2000:5000, :]
lev2_lab4_test_2cv = lev2_lab4_1.iloc[1000:2000, :]

lev2_lab5_train_2cv_11 = lev2_lab5_1.iloc[0:1000, :]
lev2_lab5_train_2cv_12 = lev2_lab5_1.iloc[2000:5000, :]
lev2_lab5_test_2cv = lev2_lab5_1.iloc[1000:2000, :]


lev2_lab6_train_2cv_11 = lev2_lab6_1.iloc[0:1000, :]
lev2_lab6_train_2cv_12 = lev2_lab6_1.iloc[2000:5000, :]
lev2_lab6_test_2cv = lev2_lab6_1.iloc[1000:2000, :]

lev2_lab7_train_2cv_11 = lev2_lab7_1.iloc[0:1000, :]
lev2_lab7_train_2cv_12 = lev2_lab7_1.iloc[2000:5000, :]
lev2_lab7_test_2cv = lev2_lab7_1.iloc[1000:2000, :]


lev2_train_2cv = pd.concat([lev2_lab0_train_2cv_11, lev2_lab0_train_2cv_12,lev2_lab1_train_2cv_11, lev2_lab1_train_2cv_12,
                          lev2_lab2_train_2cv_11, lev2_lab2_train_2cv_12,lev2_lab3_train_2cv_11, lev2_lab3_train_2cv_12,
                          lev2_lab4_train_2cv_11, lev2_lab4_train_2cv_12,lev2_lab5_train_2cv_11, lev2_lab5_train_2cv_12,
                          lev2_lab6_train_2cv_11, lev2_lab6_train_2cv_12,lev2_lab7_train_2cv_11, lev2_lab7_train_2cv_12], axis=0)
lev2_test_2cv = pd.concat([lev2_lab0_test_2cv,lev2_lab1_test_2cv,lev2_lab2_test_2cv,lev2_lab3_test_2cv,lev2_lab4_test_2cv,lev2_lab5_test_2cv,
                         lev2_lab6_test_2cv,lev2_lab7_test_2cv], axis=0)


lev2_2cv_train_X, lev2_2cv_train_Y,lev2_2cv_test_X, lev2_2cv_test_Y = load_data_det_8(lev2_train_2cv,lev2_test_2cv)





lev2_lab0_train_3cv_11 = lev2_lab0_1.iloc[0:2000, :]
lev2_lab0_train_3cv_12 = lev2_lab0_1.iloc[3000:5000, :]
lev2_lab0_test_3cv = lev2_lab0_1.iloc[2000:3000, :]

lev2_lab1_train_3cv_11 = lev2_lab1_1.iloc[0:2000, :]
lev2_lab1_train_3cv_12 = lev2_lab1_1.iloc[3000:5000, :]
lev2_lab1_test_3cv = lev2_lab1_1.iloc[2000:3000, :]

lev2_lab2_train_3cv_11 = lev2_lab2_1.iloc[0:2000, :]
lev2_lab2_train_3cv_12 = lev2_lab2_1.iloc[3000:5000, :]
lev2_lab2_test_3cv = lev2_lab2_1.iloc[2000:3000, :]

lev2_lab3_train_3cv_11 = lev2_lab3_1.iloc[0:2000, :]
lev2_lab3_train_3cv_12 = lev2_lab3_1.iloc[3000:5000, :]
lev2_lab3_test_3cv = lev2_lab3_1.iloc[2000:3000, :]

lev2_lab4_train_3cv_11 = lev2_lab4_1.iloc[0:2000, :]
lev2_lab4_train_3cv_12 = lev2_lab4_1.iloc[3000:5000, :]
lev2_lab4_test_3cv = lev2_lab4_1.iloc[2000:3000, :]

lev2_lab5_train_3cv_11 = lev2_lab5_1.iloc[0:2000, :]
lev2_lab5_train_3cv_12 = lev2_lab5_1.iloc[3000:5000, :]
lev2_lab5_test_3cv = lev2_lab5_1.iloc[2000:3000, :]


lev2_lab6_train_3cv_11 = lev2_lab6_1.iloc[0:2000, :]
lev2_lab6_train_3cv_12 = lev2_lab6_1.iloc[3000:5000, :]
lev2_lab6_test_3cv = lev2_lab6_1.iloc[2000:3000, :]

lev2_lab7_train_3cv_11 = lev2_lab7_1.iloc[0:2000, :]
lev2_lab7_train_3cv_12 = lev2_lab7_1.iloc[3000:5000, :]
lev2_lab7_test_3cv = lev2_lab7_1.iloc[2000:3000, :]


lev2_train_3cv = pd.concat([lev2_lab0_train_3cv_11, lev2_lab0_train_3cv_12,lev2_lab1_train_3cv_11, lev2_lab1_train_3cv_12,
                          lev2_lab2_train_3cv_11, lev2_lab2_train_3cv_12,lev2_lab3_train_3cv_11, lev2_lab3_train_3cv_12,
                          lev2_lab4_train_3cv_11, lev2_lab4_train_3cv_12,lev2_lab5_train_3cv_11, lev2_lab5_train_3cv_12,
                          lev2_lab6_train_3cv_11, lev2_lab6_train_3cv_12,lev2_lab7_train_3cv_11, lev2_lab7_train_3cv_12], axis=0)
lev2_test_3cv = pd.concat([lev2_lab0_test_3cv,lev2_lab1_test_3cv,lev2_lab2_test_3cv,lev2_lab3_test_3cv,lev2_lab4_test_3cv,lev2_lab5_test_3cv,
                         lev2_lab6_test_3cv,lev2_lab7_test_3cv], axis=0)


lev2_3cv_train_X, lev2_3cv_train_Y,lev2_3cv_test_X, lev2_3cv_test_Y = load_data_det_8(lev2_train_3cv,lev2_test_3cv)



lev2_lab0_train_4cv_11 = lev2_lab0_1.iloc[0:3000, :]
lev2_lab0_train_4cv_12 = lev2_lab0_1.iloc[4000:5000, :]
lev2_lab0_test_4cv = lev2_lab0_1.iloc[3000:4000, :]

lev2_lab1_train_4cv_11 = lev2_lab1_1.iloc[0:3000, :]
lev2_lab1_train_4cv_12 = lev2_lab1_1.iloc[4000:5000, :]
lev2_lab1_test_4cv = lev2_lab1_1.iloc[3000:4000, :]

lev2_lab2_train_4cv_11 = lev2_lab2_1.iloc[0:3000, :]
lev2_lab2_train_4cv_12 = lev2_lab2_1.iloc[4000:5000, :]
lev2_lab2_test_4cv = lev2_lab2_1.iloc[3000:4000, :]

lev2_lab3_train_4cv_11 = lev2_lab3_1.iloc[0:3000, :]
lev2_lab3_train_4cv_12 = lev2_lab3_1.iloc[4000:5000, :]
lev2_lab3_test_4cv = lev2_lab3_1.iloc[3000:4000, :]

lev2_lab4_train_4cv_11 = lev2_lab4_1.iloc[0:3000, :]
lev2_lab4_train_4cv_12 = lev2_lab4_1.iloc[4000:5000, :]
lev2_lab4_test_4cv = lev2_lab4_1.iloc[3000:4000, :]

lev2_lab5_train_4cv_11 = lev2_lab5_1.iloc[0:3000, :]
lev2_lab5_train_4cv_12 = lev2_lab5_1.iloc[4000:5000, :]
lev2_lab5_test_4cv = lev2_lab5_1.iloc[3000:4000, :]


lev2_lab6_train_4cv_11 = lev2_lab6_1.iloc[0:3000, :]
lev2_lab6_train_4cv_12 = lev2_lab6_1.iloc[4000:5000, :]
lev2_lab6_test_4cv = lev2_lab6_1.iloc[3000:4000, :]

lev2_lab7_train_4cv_11 = lev2_lab7_1.iloc[0:3000, :]
lev2_lab7_train_4cv_12 = lev2_lab7_1.iloc[4000:5000, :]
lev2_lab7_test_4cv = lev2_lab7_1.iloc[3000:4000, :]


lev2_train_4cv = pd.concat([lev2_lab0_train_4cv_11, lev2_lab0_train_4cv_12,lev2_lab1_train_4cv_11, lev2_lab1_train_4cv_12,
                          lev2_lab2_train_4cv_11, lev2_lab2_train_4cv_12,lev2_lab3_train_4cv_11, lev2_lab3_train_4cv_12,
                          lev2_lab4_train_4cv_11, lev2_lab4_train_4cv_12,lev2_lab5_train_4cv_11, lev2_lab5_train_4cv_12,
                          lev2_lab6_train_4cv_11, lev2_lab6_train_4cv_12,lev2_lab7_train_4cv_11, lev2_lab7_train_4cv_12], axis=0)
lev2_test_4cv = pd.concat([lev2_lab0_test_4cv,lev2_lab1_test_4cv,lev2_lab2_test_4cv,lev2_lab3_test_4cv,lev2_lab4_test_4cv,lev2_lab5_test_4cv,
                         lev2_lab6_test_4cv,lev2_lab7_test_4cv], axis=0)


lev2_4cv_train_X, lev2_4cv_train_Y,lev2_4cv_test_X, lev2_4cv_test_Y = load_data_det_8(lev2_train_4cv,lev2_test_4cv)





# 将n0个数据划分为训练集、验证集、测试集(8:1:1)
lev2_lab0_train_5cv = lev2_lab0_1.iloc[0:4000, :]
lev2_lab0_test_5cv = lev2_lab0_1.iloc[4000:5000, :]
lev2_lab1_train_5cv = lev2_lab1_1.iloc[0:4000, :]
lev2_lab1_test_5cv = lev2_lab1_1.iloc[4000:5000, :]
lev2_lab2_train_5cv = lev2_lab2_1.iloc[0:4000, :]
lev2_lab2_test_5cv = lev2_lab2_1.iloc[4000:5000, :]
lev2_lab3_train_5cv = lev2_lab3_1.iloc[0:4000, :]
lev2_lab3_test_5cv = lev2_lab3_1.iloc[4000:5000, :]
lev2_lab4_train_5cv = lev2_lab4_1.iloc[0:4000, :]
lev2_lab4_test_5cv = lev2_lab4_1.iloc[4000:5000, :]
lev2_lab5_train_5cv = lev2_lab5_1.iloc[0:4000, :]
lev2_lab5_test_5cv = lev2_lab5_1.iloc[4000:5000, :]
lev2_lab6_train_5cv = lev2_lab6_1.iloc[0:4000, :]
lev2_lab6_test_5cv = lev2_lab6_1.iloc[4000:5000, :]
lev2_lab7_train_5cv = lev2_lab7_1.iloc[0:4000, :]
lev2_lab7_test_5cv = lev2_lab7_1.iloc[4000:5000, :]

# 合并训练集测试集
lev2_train_5cv = pd.concat([lev2_lab0_train_5cv, lev2_lab1_train_5cv, lev2_lab2_train_5cv, lev2_lab3_train_5cv,
                        lev2_lab4_train_5cv, lev2_lab5_train_5cv, lev2_lab6_train_5cv, lev2_lab7_train_5cv], axis=0)
lev2_test_5cv = pd.concat([lev2_lab0_test_5cv, lev2_lab1_test_5cv, lev2_lab2_test_5cv, lev2_lab3_test_5cv,
                       lev2_lab4_test_5cv, lev2_lab5_test_5cv, lev2_lab6_test_5cv, lev2_lab7_test_5cv], axis=0)


lev2_5cv_train_X, lev2_5cv_train_Y,lev2_5cv_test_X, lev2_5cv_test_Y = load_data_det_8(lev2_train_5cv,lev2_test_5cv)




lev3_lab0 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab0.csv')
lev3_lab1 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab1.csv')
lev3_lab2 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab2.csv')
lev3_lab3 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab3.csv')
lev3_lab4 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab4.csv')
lev3_lab5 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab5.csv')
lev3_lab6 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab6.csv')
lev3_lab7 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab7.csv')

lev3_lab0 = lev3_lab0.dropna()
lev3_lab1 = lev3_lab1.dropna()
lev3_lab2 = lev3_lab2.dropna()
lev3_lab3 = lev3_lab3.dropna()
lev3_lab4 = lev3_lab4.dropna()
lev3_lab5 = lev3_lab5.dropna()
lev3_lab6 = lev3_lab6.dropna()
lev3_lab7 = lev3_lab7.dropna()

lev3_lab0_1 = lev3_lab0.iloc[0:5000, :]
lev3_lab1_1 = lev3_lab1.iloc[0:5000, :]
lev3_lab2_1 = lev3_lab2.iloc[0:5000, :]
lev3_lab3_1 = lev3_lab3.iloc[0:5000, :]
lev3_lab4_1 = lev3_lab4.iloc[0:5000, :]
lev3_lab5_1 = lev3_lab5.iloc[0:5000, :]
lev3_lab6_1 = lev3_lab6.iloc[0:5000, :]
lev3_lab7_1 = lev3_lab7.iloc[0:5000, :]



lev3_lab0_train_1cv = lev3_lab0_1.iloc[1000:5000, :]
lev3_lab0_test_1cv = lev3_lab0_1.iloc[0:1000, :]
lev3_lab1_train_1cv = lev3_lab1_1.iloc[1000:5000, :]
lev3_lab1_test_1cv = lev3_lab1_1.iloc[0:1000, :]
lev3_lab2_train_1cv = lev3_lab2_1.iloc[1000:5000, :]
lev3_lab2_test_1cv = lev3_lab2_1.iloc[0:1000, :]
lev3_lab3_train_1cv = lev3_lab3_1.iloc[1000:5000, :]
lev3_lab3_test_1cv = lev3_lab3_1.iloc[0:1000, :]
lev3_lab4_train_1cv = lev3_lab4_1.iloc[1000:5000, :]
lev3_lab4_test_1cv = lev3_lab4_1.iloc[0:1000, :]
lev3_lab5_train_1cv = lev3_lab5_1.iloc[1000:5000, :]
lev3_lab5_test_1cv = lev3_lab5_1.iloc[0:1000, :]
lev3_lab6_train_1cv = lev3_lab6_1.iloc[1000:5000, :]
lev3_lab6_test_1cv = lev3_lab6_1.iloc[0:1000, :]
lev3_lab7_train_1cv = lev3_lab7_1.iloc[1000:5000, :]
lev3_lab7_test_1cv = lev3_lab7_1.iloc[0:1000, :]

# 合并训练集测试集
lev3_train_1cv = pd.concat([lev3_lab0_train_1cv, lev3_lab1_train_1cv, lev3_lab2_train_1cv, lev3_lab3_train_1cv,
                        lev3_lab4_train_1cv, lev3_lab5_train_1cv, lev3_lab6_train_1cv, lev3_lab7_train_1cv], axis=0)
lev3_test_1cv = pd.concat([lev3_lab0_test_1cv, lev3_lab1_test_1cv, lev3_lab2_test_1cv, lev3_lab3_test_1cv,
                       lev3_lab4_test_1cv, lev3_lab5_test_1cv, lev3_lab6_test_1cv, lev3_lab7_test_1cv], axis=0)


lev3_1cv_train_X, lev3_1cv_train_Y,lev3_1cv_test_X, lev3_1cv_test_Y = load_data_det_8(lev3_train_1cv,lev3_test_1cv)






lev3_lab0_train_2cv_11 = lev3_lab0_1.iloc[0:1000, :]
lev3_lab0_train_2cv_12 = lev3_lab0_1.iloc[2000:5000, :]
lev3_lab0_test_2cv = lev3_lab0_1.iloc[1000:2000, :]

lev3_lab1_train_2cv_11 = lev3_lab1_1.iloc[0:1000, :]
lev3_lab1_train_2cv_12 = lev3_lab1_1.iloc[2000:5000, :]
lev3_lab1_test_2cv = lev3_lab1_1.iloc[1000:2000, :]

lev3_lab2_train_2cv_11 = lev3_lab2_1.iloc[0:1000, :]
lev3_lab2_train_2cv_12 = lev3_lab2_1.iloc[2000:5000, :]
lev3_lab2_test_2cv = lev3_lab2_1.iloc[1000:2000, :]

lev3_lab3_train_2cv_11 = lev3_lab3_1.iloc[0:1000, :]
lev3_lab3_train_2cv_12 = lev3_lab3_1.iloc[2000:5000, :]
lev3_lab3_test_2cv = lev3_lab3_1.iloc[1000:2000, :]

lev3_lab4_train_2cv_11 = lev3_lab4_1.iloc[0:1000, :]
lev3_lab4_train_2cv_12 = lev3_lab4_1.iloc[2000:5000, :]
lev3_lab4_test_2cv = lev3_lab4_1.iloc[1000:2000, :]

lev3_lab5_train_2cv_11 = lev3_lab5_1.iloc[0:1000, :]
lev3_lab5_train_2cv_12 = lev3_lab5_1.iloc[2000:5000, :]
lev3_lab5_test_2cv = lev3_lab5_1.iloc[1000:2000, :]


lev3_lab6_train_2cv_11 = lev3_lab6_1.iloc[0:1000, :]
lev3_lab6_train_2cv_12 = lev3_lab6_1.iloc[2000:5000, :]
lev3_lab6_test_2cv = lev3_lab6_1.iloc[1000:2000, :]

lev3_lab7_train_2cv_11 = lev3_lab7_1.iloc[0:1000, :]
lev3_lab7_train_2cv_12 = lev3_lab7_1.iloc[2000:5000, :]
lev3_lab7_test_2cv = lev3_lab7_1.iloc[1000:2000, :]


lev3_train_2cv = pd.concat([lev3_lab0_train_2cv_11, lev3_lab0_train_2cv_12,lev3_lab1_train_2cv_11, lev3_lab1_train_2cv_12,
                          lev3_lab2_train_2cv_11, lev3_lab2_train_2cv_12,lev3_lab3_train_2cv_11, lev3_lab3_train_2cv_12,
                          lev3_lab4_train_2cv_11, lev3_lab4_train_2cv_12,lev3_lab5_train_2cv_11, lev3_lab5_train_2cv_12,
                          lev3_lab6_train_2cv_11, lev3_lab6_train_2cv_12,lev3_lab7_train_2cv_11, lev3_lab7_train_2cv_12], axis=0)
lev3_test_2cv = pd.concat([lev3_lab0_test_2cv,lev3_lab1_test_2cv,lev3_lab2_test_2cv,lev3_lab3_test_2cv,lev3_lab4_test_2cv,lev3_lab5_test_2cv,
                         lev3_lab6_test_2cv,lev3_lab7_test_2cv], axis=0)


lev3_2cv_train_X, lev3_2cv_train_Y,lev3_2cv_test_X, lev3_2cv_test_Y = load_data_det_8(lev3_train_2cv,lev3_test_2cv)





lev3_lab0_train_3cv_11 = lev3_lab0_1.iloc[0:2000, :]
lev3_lab0_train_3cv_12 = lev3_lab0_1.iloc[3000:5000, :]
lev3_lab0_test_3cv = lev3_lab0_1.iloc[2000:3000, :]

lev3_lab1_train_3cv_11 = lev3_lab1_1.iloc[0:2000, :]
lev3_lab1_train_3cv_12 = lev3_lab1_1.iloc[3000:5000, :]
lev3_lab1_test_3cv = lev3_lab1_1.iloc[2000:3000, :]

lev3_lab2_train_3cv_11 = lev3_lab2_1.iloc[0:2000, :]
lev3_lab2_train_3cv_12 = lev3_lab2_1.iloc[3000:5000, :]
lev3_lab2_test_3cv = lev3_lab2_1.iloc[2000:3000, :]

lev3_lab3_train_3cv_11 = lev3_lab3_1.iloc[0:2000, :]
lev3_lab3_train_3cv_12 = lev3_lab3_1.iloc[3000:5000, :]
lev3_lab3_test_3cv = lev3_lab3_1.iloc[2000:3000, :]

lev3_lab4_train_3cv_11 = lev3_lab4_1.iloc[0:2000, :]
lev3_lab4_train_3cv_12 = lev3_lab4_1.iloc[3000:5000, :]
lev3_lab4_test_3cv = lev3_lab4_1.iloc[2000:3000, :]

lev3_lab5_train_3cv_11 = lev3_lab5_1.iloc[0:2000, :]
lev3_lab5_train_3cv_12 = lev3_lab5_1.iloc[3000:5000, :]
lev3_lab5_test_3cv = lev3_lab5_1.iloc[2000:3000, :]


lev3_lab6_train_3cv_11 = lev3_lab6_1.iloc[0:2000, :]
lev3_lab6_train_3cv_12 = lev3_lab6_1.iloc[3000:5000, :]
lev3_lab6_test_3cv = lev3_lab6_1.iloc[2000:3000, :]

lev3_lab7_train_3cv_11 = lev3_lab7_1.iloc[0:2000, :]
lev3_lab7_train_3cv_12 = lev3_lab7_1.iloc[3000:5000, :]
lev3_lab7_test_3cv = lev3_lab7_1.iloc[2000:3000, :]


lev3_train_3cv = pd.concat([lev3_lab0_train_3cv_11, lev3_lab0_train_3cv_12,lev3_lab1_train_3cv_11, lev3_lab1_train_3cv_12,
                          lev3_lab2_train_3cv_11, lev3_lab2_train_3cv_12,lev3_lab3_train_3cv_11, lev3_lab3_train_3cv_12,
                          lev3_lab4_train_3cv_11, lev3_lab4_train_3cv_12,lev3_lab5_train_3cv_11, lev3_lab5_train_3cv_12,
                          lev3_lab6_train_3cv_11, lev3_lab6_train_3cv_12,lev3_lab7_train_3cv_11, lev3_lab7_train_3cv_12], axis=0)
lev3_test_3cv = pd.concat([lev3_lab0_test_3cv,lev3_lab1_test_3cv,lev3_lab2_test_3cv,lev3_lab3_test_3cv,lev3_lab4_test_3cv,lev3_lab5_test_3cv,
                         lev3_lab6_test_3cv,lev3_lab7_test_3cv], axis=0)


lev3_3cv_train_X, lev3_3cv_train_Y,lev3_3cv_test_X, lev3_3cv_test_Y = load_data_det_8(lev3_train_3cv,lev3_test_3cv)



lev3_lab0_train_4cv_11 = lev3_lab0_1.iloc[0:3000, :]
lev3_lab0_train_4cv_12 = lev3_lab0_1.iloc[4000:5000, :]
lev3_lab0_test_4cv = lev3_lab0_1.iloc[3000:4000, :]

lev3_lab1_train_4cv_11 = lev3_lab1_1.iloc[0:3000, :]
lev3_lab1_train_4cv_12 = lev3_lab1_1.iloc[4000:5000, :]
lev3_lab1_test_4cv = lev3_lab1_1.iloc[3000:4000, :]

lev3_lab2_train_4cv_11 = lev3_lab2_1.iloc[0:3000, :]
lev3_lab2_train_4cv_12 = lev3_lab2_1.iloc[4000:5000, :]
lev3_lab2_test_4cv = lev3_lab2_1.iloc[3000:4000, :]

lev3_lab3_train_4cv_11 = lev3_lab3_1.iloc[0:3000, :]
lev3_lab3_train_4cv_12 = lev3_lab3_1.iloc[4000:5000, :]
lev3_lab3_test_4cv = lev3_lab3_1.iloc[3000:4000, :]

lev3_lab4_train_4cv_11 = lev3_lab4_1.iloc[0:3000, :]
lev3_lab4_train_4cv_12 = lev3_lab4_1.iloc[4000:5000, :]
lev3_lab4_test_4cv = lev3_lab4_1.iloc[3000:4000, :]

lev3_lab5_train_4cv_11 = lev3_lab5_1.iloc[0:3000, :]
lev3_lab5_train_4cv_12 = lev3_lab5_1.iloc[4000:5000, :]
lev3_lab5_test_4cv = lev3_lab5_1.iloc[3000:4000, :]


lev3_lab6_train_4cv_11 = lev3_lab6_1.iloc[0:3000, :]
lev3_lab6_train_4cv_12 = lev3_lab6_1.iloc[4000:5000, :]
lev3_lab6_test_4cv = lev3_lab6_1.iloc[3000:4000, :]

lev3_lab7_train_4cv_11 = lev3_lab7_1.iloc[0:3000, :]
lev3_lab7_train_4cv_12 = lev3_lab7_1.iloc[4000:5000, :]
lev3_lab7_test_4cv = lev3_lab7_1.iloc[3000:4000, :]


lev3_train_4cv = pd.concat([lev3_lab0_train_4cv_11, lev3_lab0_train_4cv_12,lev3_lab1_train_4cv_11, lev3_lab1_train_4cv_12,
                          lev3_lab2_train_4cv_11, lev3_lab2_train_4cv_12,lev3_lab3_train_4cv_11, lev3_lab3_train_4cv_12,
                          lev3_lab4_train_4cv_11, lev3_lab4_train_4cv_12,lev3_lab5_train_4cv_11, lev3_lab5_train_4cv_12,
                          lev3_lab6_train_4cv_11, lev3_lab6_train_4cv_12,lev3_lab7_train_4cv_11, lev3_lab7_train_4cv_12], axis=0)
lev3_test_4cv = pd.concat([lev3_lab0_test_4cv,lev3_lab1_test_4cv,lev3_lab2_test_4cv,lev3_lab3_test_4cv,lev3_lab4_test_4cv,lev3_lab5_test_4cv,
                         lev3_lab6_test_4cv,lev3_lab7_test_4cv], axis=0)


lev3_4cv_train_X, lev3_4cv_train_Y,lev3_4cv_test_X, lev3_4cv_test_Y = load_data_det_8(lev3_train_4cv,lev3_test_4cv)





# 将n0个数据划分为训练集、验证集、测试集(8:1:1)
lev3_lab0_train_5cv = lev3_lab0_1.iloc[0:4000, :]
lev3_lab0_test_5cv = lev3_lab0_1.iloc[4000:5000, :]
lev3_lab1_train_5cv = lev3_lab1_1.iloc[0:4000, :]
lev3_lab1_test_5cv = lev3_lab1_1.iloc[4000:5000, :]
lev3_lab2_train_5cv = lev3_lab2_1.iloc[0:4000, :]
lev3_lab2_test_5cv = lev3_lab2_1.iloc[4000:5000, :]
lev3_lab3_train_5cv = lev3_lab3_1.iloc[0:4000, :]
lev3_lab3_test_5cv = lev3_lab3_1.iloc[4000:5000, :]
lev3_lab4_train_5cv = lev3_lab4_1.iloc[0:4000, :]
lev3_lab4_test_5cv = lev3_lab4_1.iloc[4000:5000, :]
lev3_lab5_train_5cv = lev3_lab5_1.iloc[0:4000, :]
lev3_lab5_test_5cv = lev3_lab5_1.iloc[4000:5000, :]
lev3_lab6_train_5cv = lev3_lab6_1.iloc[0:4000, :]
lev3_lab6_test_5cv = lev3_lab6_1.iloc[4000:5000, :]
lev3_lab7_train_5cv = lev3_lab7_1.iloc[0:4000, :]
lev3_lab7_test_5cv = lev3_lab7_1.iloc[4000:5000, :]

# 合并训练集测试集
lev3_train_5cv = pd.concat([lev3_lab0_train_5cv, lev3_lab1_train_5cv, lev3_lab2_train_5cv, lev3_lab3_train_5cv,
                        lev3_lab4_train_5cv, lev3_lab5_train_5cv, lev3_lab6_train_5cv, lev3_lab7_train_5cv], axis=0)
lev3_test_5cv = pd.concat([lev3_lab0_test_5cv, lev3_lab1_test_5cv, lev3_lab2_test_5cv, lev3_lab3_test_5cv,
                       lev3_lab4_test_5cv, lev3_lab5_test_5cv, lev3_lab6_test_5cv, lev3_lab7_test_5cv], axis=0)


lev3_5cv_train_X, lev3_5cv_train_Y,lev3_5cv_test_X, lev3_5cv_test_Y = load_data_det_8(lev3_train_5cv,lev3_test_5cv)




lev4_lab0 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab0.csv')
lev4_lab1 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab1.csv')
lev4_lab2 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab2.csv')
lev4_lab3 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab3.csv')
lev4_lab4 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab4.csv')
lev4_lab5 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab5.csv')
lev4_lab6 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab6.csv')
lev4_lab7 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab7.csv')

lev4_lab0 = lev4_lab0.dropna()
lev4_lab1 = lev4_lab1.dropna()
lev4_lab2 = lev4_lab2.dropna()
lev4_lab3 = lev4_lab3.dropna()
lev4_lab4 = lev4_lab4.dropna()
lev4_lab5 = lev4_lab5.dropna()
lev4_lab6 = lev4_lab6.dropna()
lev4_lab7 = lev4_lab7.dropna()

lev4_lab0_1 = lev4_lab0.iloc[0:5000, :]
lev4_lab1_1 = lev4_lab1.iloc[0:5000, :]
lev4_lab2_1 = lev4_lab2.iloc[0:5000, :]
lev4_lab3_1 = lev4_lab3.iloc[0:5000, :]
lev4_lab4_1 = lev4_lab4.iloc[0:5000, :]
lev4_lab5_1 = lev4_lab5.iloc[0:5000, :]
lev4_lab6_1 = lev4_lab6.iloc[0:5000, :]
lev4_lab7_1 = lev4_lab7.iloc[0:5000, :]



lev4_lab0_train_1cv = lev4_lab0_1.iloc[1000:5000, :]
lev4_lab0_test_1cv = lev4_lab0_1.iloc[0:1000, :]
lev4_lab1_train_1cv = lev4_lab1_1.iloc[1000:5000, :]
lev4_lab1_test_1cv = lev4_lab1_1.iloc[0:1000, :]
lev4_lab2_train_1cv = lev4_lab2_1.iloc[1000:5000, :]
lev4_lab2_test_1cv = lev4_lab2_1.iloc[0:1000, :]
lev4_lab3_train_1cv = lev4_lab3_1.iloc[1000:5000, :]
lev4_lab3_test_1cv = lev4_lab3_1.iloc[0:1000, :]
lev4_lab4_train_1cv = lev4_lab4_1.iloc[1000:5000, :]
lev4_lab4_test_1cv = lev4_lab4_1.iloc[0:1000, :]
lev4_lab5_train_1cv = lev4_lab5_1.iloc[1000:5000, :]
lev4_lab5_test_1cv = lev4_lab5_1.iloc[0:1000, :]
lev4_lab6_train_1cv = lev4_lab6_1.iloc[1000:5000, :]
lev4_lab6_test_1cv = lev4_lab6_1.iloc[0:1000, :]
lev4_lab7_train_1cv = lev4_lab7_1.iloc[1000:5000, :]
lev4_lab7_test_1cv = lev4_lab7_1.iloc[0:1000, :]

# 合并训练集测试集
lev4_train_1cv = pd.concat([lev4_lab0_train_1cv, lev4_lab1_train_1cv, lev4_lab2_train_1cv, lev4_lab3_train_1cv,
                        lev4_lab4_train_1cv, lev4_lab5_train_1cv, lev4_lab6_train_1cv, lev4_lab7_train_1cv], axis=0)
lev4_test_1cv = pd.concat([lev4_lab0_test_1cv, lev4_lab1_test_1cv, lev4_lab2_test_1cv, lev4_lab3_test_1cv,
                       lev4_lab4_test_1cv, lev4_lab5_test_1cv, lev4_lab6_test_1cv, lev4_lab7_test_1cv], axis=0)


lev4_1cv_train_X, lev4_1cv_train_Y,lev4_1cv_test_X, lev4_1cv_test_Y = load_data_det_8(lev4_train_1cv,lev4_test_1cv)






lev4_lab0_train_2cv_11 = lev4_lab0_1.iloc[0:1000, :]
lev4_lab0_train_2cv_12 = lev4_lab0_1.iloc[2000:5000, :]
lev4_lab0_test_2cv = lev4_lab0_1.iloc[1000:2000, :]

lev4_lab1_train_2cv_11 = lev4_lab1_1.iloc[0:1000, :]
lev4_lab1_train_2cv_12 = lev4_lab1_1.iloc[2000:5000, :]
lev4_lab1_test_2cv = lev4_lab1_1.iloc[1000:2000, :]

lev4_lab2_train_2cv_11 = lev4_lab2_1.iloc[0:1000, :]
lev4_lab2_train_2cv_12 = lev4_lab2_1.iloc[2000:5000, :]
lev4_lab2_test_2cv = lev4_lab2_1.iloc[1000:2000, :]

lev4_lab3_train_2cv_11 = lev4_lab3_1.iloc[0:1000, :]
lev4_lab3_train_2cv_12 = lev4_lab3_1.iloc[2000:5000, :]
lev4_lab3_test_2cv = lev4_lab3_1.iloc[1000:2000, :]

lev4_lab4_train_2cv_11 = lev4_lab4_1.iloc[0:1000, :]
lev4_lab4_train_2cv_12 = lev4_lab4_1.iloc[2000:5000, :]
lev4_lab4_test_2cv = lev4_lab4_1.iloc[1000:2000, :]

lev4_lab5_train_2cv_11 = lev4_lab5_1.iloc[0:1000, :]
lev4_lab5_train_2cv_12 = lev4_lab5_1.iloc[2000:5000, :]
lev4_lab5_test_2cv = lev4_lab5_1.iloc[1000:2000, :]


lev4_lab6_train_2cv_11 = lev4_lab6_1.iloc[0:1000, :]
lev4_lab6_train_2cv_12 = lev4_lab6_1.iloc[2000:5000, :]
lev4_lab6_test_2cv = lev4_lab6_1.iloc[1000:2000, :]

lev4_lab7_train_2cv_11 = lev4_lab7_1.iloc[0:1000, :]
lev4_lab7_train_2cv_12 = lev4_lab7_1.iloc[2000:5000, :]
lev4_lab7_test_2cv = lev4_lab7_1.iloc[1000:2000, :]


lev4_train_2cv = pd.concat([lev4_lab0_train_2cv_11, lev4_lab0_train_2cv_12,lev4_lab1_train_2cv_11, lev4_lab1_train_2cv_12,
                          lev4_lab2_train_2cv_11, lev4_lab2_train_2cv_12,lev4_lab3_train_2cv_11, lev4_lab3_train_2cv_12,
                          lev4_lab4_train_2cv_11, lev4_lab4_train_2cv_12,lev4_lab5_train_2cv_11, lev4_lab5_train_2cv_12,
                          lev4_lab6_train_2cv_11, lev4_lab6_train_2cv_12,lev4_lab7_train_2cv_11, lev4_lab7_train_2cv_12], axis=0)
lev4_test_2cv = pd.concat([lev4_lab0_test_2cv,lev4_lab1_test_2cv,lev4_lab2_test_2cv,lev4_lab3_test_2cv,lev4_lab4_test_2cv,lev4_lab5_test_2cv,
                         lev4_lab6_test_2cv,lev4_lab7_test_2cv], axis=0)


lev4_2cv_train_X, lev4_2cv_train_Y,lev4_2cv_test_X, lev4_2cv_test_Y = load_data_det_8(lev4_train_2cv,lev4_test_2cv)





lev4_lab0_train_3cv_11 = lev4_lab0_1.iloc[0:2000, :]
lev4_lab0_train_3cv_12 = lev4_lab0_1.iloc[3000:5000, :]
lev4_lab0_test_3cv = lev4_lab0_1.iloc[2000:3000, :]

lev4_lab1_train_3cv_11 = lev4_lab1_1.iloc[0:2000, :]
lev4_lab1_train_3cv_12 = lev4_lab1_1.iloc[3000:5000, :]
lev4_lab1_test_3cv = lev4_lab1_1.iloc[2000:3000, :]

lev4_lab2_train_3cv_11 = lev4_lab2_1.iloc[0:2000, :]
lev4_lab2_train_3cv_12 = lev4_lab2_1.iloc[3000:5000, :]
lev4_lab2_test_3cv = lev4_lab2_1.iloc[2000:3000, :]

lev4_lab3_train_3cv_11 = lev4_lab3_1.iloc[0:2000, :]
lev4_lab3_train_3cv_12 = lev4_lab3_1.iloc[3000:5000, :]
lev4_lab3_test_3cv = lev4_lab3_1.iloc[2000:3000, :]

lev4_lab4_train_3cv_11 = lev4_lab4_1.iloc[0:2000, :]
lev4_lab4_train_3cv_12 = lev4_lab4_1.iloc[3000:5000, :]
lev4_lab4_test_3cv = lev4_lab4_1.iloc[2000:3000, :]

lev4_lab5_train_3cv_11 = lev4_lab5_1.iloc[0:2000, :]
lev4_lab5_train_3cv_12 = lev4_lab5_1.iloc[3000:5000, :]
lev4_lab5_test_3cv = lev4_lab5_1.iloc[2000:3000, :]


lev4_lab6_train_3cv_11 = lev4_lab6_1.iloc[0:2000, :]
lev4_lab6_train_3cv_12 = lev4_lab6_1.iloc[3000:5000, :]
lev4_lab6_test_3cv = lev4_lab6_1.iloc[2000:3000, :]

lev4_lab7_train_3cv_11 = lev4_lab7_1.iloc[0:2000, :]
lev4_lab7_train_3cv_12 = lev4_lab7_1.iloc[3000:5000, :]
lev4_lab7_test_3cv = lev4_lab7_1.iloc[2000:3000, :]


lev4_train_3cv = pd.concat([lev4_lab0_train_3cv_11, lev4_lab0_train_3cv_12,lev4_lab1_train_3cv_11, lev4_lab1_train_3cv_12,
                          lev4_lab2_train_3cv_11, lev4_lab2_train_3cv_12,lev4_lab3_train_3cv_11, lev4_lab3_train_3cv_12,
                          lev4_lab4_train_3cv_11, lev4_lab4_train_3cv_12,lev4_lab5_train_3cv_11, lev4_lab5_train_3cv_12,
                          lev4_lab6_train_3cv_11, lev4_lab6_train_3cv_12,lev4_lab7_train_3cv_11, lev4_lab7_train_3cv_12], axis=0)
lev4_test_3cv = pd.concat([lev4_lab0_test_3cv,lev4_lab1_test_3cv,lev4_lab2_test_3cv,lev4_lab3_test_3cv,lev4_lab4_test_3cv,lev4_lab5_test_3cv,
                         lev4_lab6_test_3cv,lev4_lab7_test_3cv], axis=0)


lev4_3cv_train_X, lev4_3cv_train_Y,lev4_3cv_test_X, lev4_3cv_test_Y = load_data_det_8(lev4_train_3cv,lev4_test_3cv)



lev4_lab0_train_4cv_11 = lev4_lab0_1.iloc[0:3000, :]
lev4_lab0_train_4cv_12 = lev4_lab0_1.iloc[4000:5000, :]
lev4_lab0_test_4cv = lev4_lab0_1.iloc[3000:4000, :]

lev4_lab1_train_4cv_11 = lev4_lab1_1.iloc[0:3000, :]
lev4_lab1_train_4cv_12 = lev4_lab1_1.iloc[4000:5000, :]
lev4_lab1_test_4cv = lev4_lab1_1.iloc[3000:4000, :]

lev4_lab2_train_4cv_11 = lev4_lab2_1.iloc[0:3000, :]
lev4_lab2_train_4cv_12 = lev4_lab2_1.iloc[4000:5000, :]
lev4_lab2_test_4cv = lev4_lab2_1.iloc[3000:4000, :]

lev4_lab3_train_4cv_11 = lev4_lab3_1.iloc[0:3000, :]
lev4_lab3_train_4cv_12 = lev4_lab3_1.iloc[4000:5000, :]
lev4_lab3_test_4cv = lev4_lab3_1.iloc[3000:4000, :]

lev4_lab4_train_4cv_11 = lev4_lab4_1.iloc[0:3000, :]
lev4_lab4_train_4cv_12 = lev4_lab4_1.iloc[4000:5000, :]
lev4_lab4_test_4cv = lev4_lab4_1.iloc[3000:4000, :]

lev4_lab5_train_4cv_11 = lev4_lab5_1.iloc[0:3000, :]
lev4_lab5_train_4cv_12 = lev4_lab5_1.iloc[4000:5000, :]
lev4_lab5_test_4cv = lev4_lab5_1.iloc[3000:4000, :]


lev4_lab6_train_4cv_11 = lev4_lab6_1.iloc[0:3000, :]
lev4_lab6_train_4cv_12 = lev4_lab6_1.iloc[4000:5000, :]
lev4_lab6_test_4cv = lev4_lab6_1.iloc[3000:4000, :]

lev4_lab7_train_4cv_11 = lev4_lab7_1.iloc[0:3000, :]
lev4_lab7_train_4cv_12 = lev4_lab7_1.iloc[4000:5000, :]
lev4_lab7_test_4cv = lev4_lab7_1.iloc[3000:4000, :]


lev4_train_4cv = pd.concat([lev4_lab0_train_4cv_11, lev4_lab0_train_4cv_12,lev4_lab1_train_4cv_11, lev4_lab1_train_4cv_12,
                          lev4_lab2_train_4cv_11, lev4_lab2_train_4cv_12,lev4_lab3_train_4cv_11, lev4_lab3_train_4cv_12,
                          lev4_lab4_train_4cv_11, lev4_lab4_train_4cv_12,lev4_lab5_train_4cv_11, lev4_lab5_train_4cv_12,
                          lev4_lab6_train_4cv_11, lev4_lab6_train_4cv_12,lev4_lab7_train_4cv_11, lev4_lab7_train_4cv_12], axis=0)
lev4_test_4cv = pd.concat([lev4_lab0_test_4cv,lev4_lab1_test_4cv,lev4_lab2_test_4cv,lev4_lab3_test_4cv,lev4_lab4_test_4cv,lev4_lab5_test_4cv,
                         lev4_lab6_test_4cv,lev4_lab7_test_4cv], axis=0)


lev4_4cv_train_X, lev4_4cv_train_Y,lev4_4cv_test_X, lev4_4cv_test_Y = load_data_det_8(lev4_train_4cv,lev4_test_4cv)





# 将n0个数据划分为训练集、验证集、测试集(8:1:1)
lev4_lab0_train_5cv = lev4_lab0_1.iloc[0:4000, :]
lev4_lab0_test_5cv = lev4_lab0_1.iloc[4000:5000, :]
lev4_lab1_train_5cv = lev4_lab1_1.iloc[0:4000, :]
lev4_lab1_test_5cv = lev4_lab1_1.iloc[4000:5000, :]
lev4_lab2_train_5cv = lev4_lab2_1.iloc[0:4000, :]
lev4_lab2_test_5cv = lev4_lab2_1.iloc[4000:5000, :]
lev4_lab3_train_5cv = lev4_lab3_1.iloc[0:4000, :]
lev4_lab3_test_5cv = lev4_lab3_1.iloc[4000:5000, :]
lev4_lab4_train_5cv = lev4_lab4_1.iloc[0:4000, :]
lev4_lab4_test_5cv = lev4_lab4_1.iloc[4000:5000, :]
lev4_lab5_train_5cv = lev4_lab5_1.iloc[0:4000, :]
lev4_lab5_test_5cv = lev4_lab5_1.iloc[4000:5000, :]
lev4_lab6_train_5cv = lev4_lab6_1.iloc[0:4000, :]
lev4_lab6_test_5cv = lev4_lab6_1.iloc[4000:5000, :]
lev4_lab7_train_5cv = lev4_lab7_1.iloc[0:4000, :]
lev4_lab7_test_5cv = lev4_lab7_1.iloc[4000:5000, :]

# 合并训练集测试集
lev4_train_5cv = pd.concat([lev4_lab0_train_5cv, lev4_lab1_train_5cv, lev4_lab2_train_5cv, lev4_lab3_train_5cv,
                        lev4_lab4_train_5cv, lev4_lab5_train_5cv, lev4_lab6_train_5cv, lev4_lab7_train_5cv], axis=0)
lev4_test_5cv = pd.concat([lev4_lab0_test_5cv, lev4_lab1_test_5cv, lev4_lab2_test_5cv, lev4_lab3_test_5cv,
                       lev4_lab4_test_5cv, lev4_lab5_test_5cv, lev4_lab6_test_5cv, lev4_lab7_test_5cv], axis=0)


lev4_5cv_train_X, lev4_5cv_train_Y,lev4_5cv_test_X, lev4_5cv_test_Y = load_data_det_8(lev4_train_5cv,lev4_test_5cv)