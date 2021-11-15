import numpy as np
import pandas as pd

def load_data_det_8(train_data, val_data):
    #########钟超文的特征选择
    train_X = train_data[
        ['SF-WAT', 'SA-CFM', 'RA-CFM', 'SA-TEMP', 'MA-TEMP', 'RA-TEMP', 'SA-HUMD', 'RA-HUMD', 'OA-TEMP', 'HWC-EWT',
         'E_ccoil']]
    val_X = val_data[
        ['SF-WAT', 'SA-CFM', 'RA-CFM', 'SA-TEMP', 'MA-TEMP', 'RA-TEMP', 'SA-HUMD', 'RA-HUMD', 'OA-TEMP', 'HWC-EWT',
         'E_ccoil']]
    train_X = train_X.values
    val_X = val_X.values

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], 1)

    train_Y = train_data.iloc[:, 0]
    val_Y = val_data.iloc[:, 0]

    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    train_Y = train_Y.reshape(-1, 1)
    val_Y = val_Y.reshape(-1, 1)

    return train_X, train_Y, val_X, val_Y


lab0 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0820A.csv')
lab1 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0821A.csv')
lab2 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0822A.csv')
lab3 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0823A.csv')
lab4 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0824A.csv')
lab5 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0826A.csv')
lab6 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0827A.csv')
lab7 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0831A.csv')
lab8 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0901A.csv')
lab9 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0902A.csv')
lab10 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0906A.csv')
lab11 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0907A.csv')
lab12 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0908A.csv')


lab0_1 = lab0.iloc[360:1081, :]
lab1_1 = lab1.iloc[360:1081, :]
lab2_1 = lab2.iloc[360:1081, :]
lab3_1 = lab3.iloc[360:1081, :]
lab4_1 = lab4.iloc[360:1081, :]
lab5_1 = lab5.iloc[360:1081, :]
lab6_1 = lab6.iloc[360:1081, :]
lab7_1 = lab7.iloc[360:1081, :]
lab8_1 = lab8.iloc[360:1081, :]
lab9_1 = lab9.iloc[360:1081, :]
lab10_1 = lab10.iloc[360:1081, :]
lab11_1 = lab11.iloc[360:1081, :]
lab12_1 = lab12.iloc[360:1081, :]

lab0_2 = lab0_1.iloc[0:720, :]
lab1_2 = lab1_1.iloc[0:720, :]
lab2_2 = lab2_1.iloc[0:720, :]
lab3_2 = lab3_1.iloc[0:720, :]
lab4_2 = lab4_1.iloc[0:720, :]
lab5_2 = lab5_1.iloc[0:720, :]
lab6_2 = lab6_1.iloc[0:720, :]
lab7_2 = lab7_1.iloc[0:720, :]
lab8_2 = lab8_1.iloc[0:720, :]
lab9_2 = lab9_1.iloc[0:720, :]
lab10_2 = lab10_1.iloc[0:720, :]
lab11_2 = lab11_1.iloc[0:720, :]
lab12_2 = lab12_1.iloc[0:720, :]



summer_lab0_train_4cv_11 = lab0_2.iloc[0:432, :]
summer_lab0_train_4cv_12 = lab0_2.iloc[576:720, :]
summer_lab0_test_4cv = lab0_2.iloc[432:576, :]

summer_lab1_train_4cv_11 = lab1_2.iloc[0:432, :]
summer_lab1_train_4cv_12 = lab1_2.iloc[576:720, :]
summer_lab1_test_4cv = lab1_2.iloc[432:576, :]

summer_lab2_train_4cv_11 = lab2_2.iloc[0:432, :]
summer_lab2_train_4cv_12 = lab2_2.iloc[576:720, :]
summer_lab2_test_4cv = lab2_2.iloc[432:576, :]

summer_lab3_train_4cv_11 = lab3_2.iloc[0:432, :]
summer_lab3_train_4cv_12 = lab3_2.iloc[576:720, :]
summer_lab3_test_4cv = lab3_2.iloc[432:576, :]

summer_lab4_train_4cv_11 = lab4_2.iloc[0:432, :]
summer_lab4_train_4cv_12 = lab4_2.iloc[576:720, :]
summer_lab4_test_4cv = lab4_2.iloc[432:576, :]

summer_lab5_train_4cv_11 = lab5_2.iloc[0:432, :]
summer_lab5_train_4cv_12 = lab5_2.iloc[576:720, :]
summer_lab5_test_4cv = lab5_2.iloc[432:576, :]


summer_lab6_train_4cv_11 = lab6_2.iloc[0:432, :]
summer_lab6_train_4cv_12 = lab6_2.iloc[576:720, :]
summer_lab6_test_4cv = lab6_2.iloc[432:576, :]

summer_lab7_train_4cv_11 = lab7_2.iloc[0:432, :]
summer_lab7_train_4cv_12 = lab7_2.iloc[576:720, :]
summer_lab7_test_4cv = lab7_2.iloc[432:576, :]


summer_lab8_train_4cv_11 = lab8_2.iloc[0:432, :]
summer_lab8_train_4cv_12 = lab8_2.iloc[576:720, :]
summer_lab8_test_4cv = lab8_2.iloc[432:576, :]


summer_lab9_train_4cv_11 = lab9_2.iloc[0:432, :]
summer_lab9_train_4cv_12 = lab9_2.iloc[576:720, :]
summer_lab9_test_4cv = lab9_2.iloc[432:576, :]



summer_lab10_train_4cv_11 = lab10_2.iloc[0:432, :]
summer_lab10_train_4cv_12 = lab10_2.iloc[576:720, :]
summer_lab10_test_4cv = lab10_2.iloc[432:576, :]



summer_lab11_train_4cv_11 = lab11_2.iloc[0:432, :]
summer_lab11_train_4cv_12 = lab11_2.iloc[576:720, :]
summer_lab11_test_4cv = lab11_2.iloc[432:576, :]



summer_lab12_train_4cv_11 = lab12_2.iloc[0:432, :]
summer_lab12_train_4cv_12 = lab12_2.iloc[576:720, :]
summer_lab12_test_4cv = lab12_2.iloc[432:576, :]





summer_train_4cv = pd.concat([summer_lab0_train_4cv_11, summer_lab0_train_4cv_12,summer_lab1_train_4cv_11, summer_lab1_train_4cv_12,
                          summer_lab2_train_4cv_11, summer_lab2_train_4cv_12,summer_lab3_train_4cv_11, summer_lab3_train_4cv_12,
                          summer_lab4_train_4cv_11, summer_lab4_train_4cv_12,summer_lab5_train_4cv_11, summer_lab5_train_4cv_12,
                          summer_lab6_train_4cv_11, summer_lab6_train_4cv_12,summer_lab7_train_4cv_11, summer_lab7_train_4cv_12,
                          summer_lab8_train_4cv_11, summer_lab8_train_4cv_12,summer_lab9_train_4cv_11, summer_lab9_train_4cv_12,
                          summer_lab10_train_4cv_11, summer_lab10_train_4cv_12,summer_lab11_train_4cv_11, summer_lab11_train_4cv_12,
                          summer_lab12_train_4cv_11, summer_lab12_train_4cv_12,], axis=0)
summer_test_4cv = pd.concat([summer_lab0_test_4cv,summer_lab1_test_4cv,summer_lab2_test_4cv,summer_lab3_test_4cv,summer_lab4_test_4cv,summer_lab5_test_4cv,
                         summer_lab6_test_4cv,summer_lab7_test_4cv,summer_lab8_test_4cv,summer_lab9_test_4cv,summer_lab10_test_4cv,summer_lab11_test_4cv,summer_lab12_test_4cv], axis=0)

summer_4cv_train_X, summer_4cv_train_Y, summer_4cv_test_X, summer_4cv_test_Y = load_data_det_8(summer_train_4cv,summer_test_4cv)