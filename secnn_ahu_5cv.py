#########################选取500个    2000-1500   8+1.5+0.5
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers

from summer_data_1cv import summer_1cv_train_X, summer_1cv_train_Y,summer_1cv_test_X, summer_1cv_test_Y
from summer_data_2cv import summer_2cv_train_X, summer_2cv_train_Y,summer_2cv_test_X, summer_2cv_test_Y
from summer_data_3cv import summer_3cv_train_X, summer_3cv_train_Y,summer_3cv_test_X, summer_3cv_test_Y
from summer_data_4cv import summer_4cv_train_X, summer_4cv_train_Y,summer_4cv_test_X, summer_4cv_test_Y
from summer_data_5cv import summer_5cv_train_X, summer_5cv_train_Y,summer_5cv_test_X, summer_5cv_test_Y

min_max_scaler = MinMaxScaler()
# np.random.seed(6)
np.random.seed(6)
o = 1
p = 1
patienc=30
f0 = []
f1 = []
f2 = []
f3 = []
f4 = []
f5 = []
f6 = []
f7 = []
f8 = []
f9 = []
f10 = []
f11 = []
f12 = []
f_ar = []
acc = []

def cnnse():

    input_deep = tf.keras.layers.Input(shape=(11, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    # hidden2 = tf.keras.layers.Conv1D(filters = 32,kernel_size = 3,padding = 'same',activation = 'relu')(hidden1)
    x = tf.keras.layers.GlobalAvgPool1D()(hidden1)
    x = tf.keras.layers.Dense(int(x.shape[-1]) // 8, activation='relu')(x)
    x = tf.keras.layers.Dense(int(hidden1.shape[-1]), activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([hidden1, x])
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    xx = tf.keras.layers.GlobalAvgPool1D()(hidden4)
    xx = tf.keras.layers.Dense(int(xx.shape[-1]) // 8, activation='relu')(xx)
    xx = tf.keras.layers.Dense(int(hidden4.shape[-1]), activation='sigmoid')(xx)
    xx = tf.keras.layers.Multiply()([hidden4, xx])
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(xx)
    hidden10 = tf.keras.layers.Flatten()(hidden7)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    dp = tf.keras.layers.Dropout(0.2)(hidden111)
    ###全连接层
    output = tf.keras.layers.Dense(13, activation='softmax')(dp)

    cnn_se_summer_2000 = tf.keras.models.Model(inputs=input_deep,outputs=[output])
    cnn_se_summer_2000_det1 = cnn_se_summer_2000
    cnn_se_summer_2000_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])

    return cnn_se_summer_2000_det1
# cnn_se_summer_2000_det1.summary()

time1 = time.time()

###一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合keras.callbacks.EarlyStopping就是用来提前结束训练的。
callback1 = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50, restore_best_weights=True)]
for i in range(0,5):
    model1 = cnnse()
    cnn_se_summer_1cv =model1.fit(summer_1cv_train_X, summer_1cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
    det_cnn_se_summer_1cv = model1.predict(summer_1cv_test_X)
    a_cnn_se_summer_1cv = np.argmax(det_cnn_se_summer_1cv, axis=1)
    a_cnn_se_summer_1cv =  a_cnn_se_summer_1cv.reshape(-1, 1)
    # 输出总的的故障检测与分类精度
    a_cnn_se_summer_1cv_AC = accuracy_score(summer_1cv_test_Y, a_cnn_se_summer_1cv)
    acc.append(a_cnn_se_summer_1cv_AC)
    level1_cfm = confusion_matrix(summer_1cv_test_Y, a_cnn_se_summer_1cv)
    level1_conf = pd.DataFrame(level1_cfm).transpose()
    level1_conf.to_excel("secnnlevel1_1cv_"+str(i)+"confusion_matrix.xlsx", index=True)
    print(level1_cfm)
    level1_report = classification_report(summer_1cv_test_Y, a_cnn_se_summer_1cv, output_dict=True)
    level1_df = pd.DataFrame(level1_report).transpose()
    level1_df.to_excel("secnnlevel1_1cv_"+str(i)+"classification_report.xlsx", index=True)

    print('a_cnn_se_summer_1cv_AC=', a_cnn_se_summer_1cv_AC)
    fault0_1cv = accuracy_score(summer_1cv_test_Y[:144], a_cnn_se_summer_1cv[:144])
    fault1_1cv = accuracy_score(summer_1cv_test_Y[144:288], a_cnn_se_summer_1cv[144:288])
    fault2_1cv = accuracy_score(summer_1cv_test_Y[288:432], a_cnn_se_summer_1cv[288:432])
    fault3_1cv = accuracy_score(summer_1cv_test_Y[432:576], a_cnn_se_summer_1cv[432:576])
    fault4_1cv = accuracy_score(summer_1cv_test_Y[576:720], a_cnn_se_summer_1cv[576:720])
    fault5_1cv = accuracy_score(summer_1cv_test_Y[720:864], a_cnn_se_summer_1cv[720:864])
    fault6_1cv = accuracy_score(summer_1cv_test_Y[864:1008], a_cnn_se_summer_1cv[864:1008])
    fault7_1cv = accuracy_score(summer_1cv_test_Y[1008:1152], a_cnn_se_summer_1cv[1008:1152])
    fault8_1cv = accuracy_score(summer_1cv_test_Y[1152:1296], a_cnn_se_summer_1cv[1152:1296])
    fault9_1cv = accuracy_score(summer_1cv_test_Y[1296:1440], a_cnn_se_summer_1cv[1296:1440])
    fault10_1cv = accuracy_score(summer_1cv_test_Y[1440:1584], a_cnn_se_summer_1cv[1440:1584])
    fault11_1cv = accuracy_score(summer_1cv_test_Y[1584:1728], a_cnn_se_summer_1cv[1584:1728])
    fault12_1cv = accuracy_score(summer_1cv_test_Y[1728:1872], a_cnn_se_summer_1cv[1728:1872])

    f0.append(fault0_1cv)
    f1.append(fault1_1cv)
    f2.append(fault2_1cv)
    f3.append(fault3_1cv)
    f4.append(fault4_1cv)
    f5.append(fault5_1cv)
    f6.append(fault6_1cv)
    f7.append(fault7_1cv)
    f8.append(fault8_1cv)
    f9.append(fault9_1cv)
    f10.append(fault10_1cv)
    f11.append(fault11_1cv)
    f12.append(fault12_1cv)
    print('\n\n\n')
    print(acc)
#



for i in range(0,5):
    model2 = cnnse()
    cnn_se_summer_2cv =model2.fit(summer_2cv_train_X, summer_2cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
    det_cnn_se_summer_2cv = model2.predict(summer_2cv_test_X)
    a_cnn_se_summer_2cv = np.argmax(det_cnn_se_summer_2cv, axis=1)
    a_cnn_se_summer_2cv =  a_cnn_se_summer_2cv.reshape(-1, 1)
    # 输出总的的故障检测与分类精度
    a_cnn_se_summer_2cv_AC = accuracy_score(summer_2cv_test_Y, a_cnn_se_summer_2cv)
    acc.append(a_cnn_se_summer_2cv_AC)
    level1_cfm = confusion_matrix(summer_2cv_test_Y, a_cnn_se_summer_2cv)
    level1_conf = pd.DataFrame(level1_cfm).transpose()
    # level1_conf.to_excel("level1_2cv_confusion_matrix.xlsx", index=True)
    level1_conf.to_excel("secnnlevel1_2cv_" + str(i) + "confusion_matrix.xlsx", index=True)
    print(level1_cfm)
    level1_report = classification_report(summer_2cv_test_Y, a_cnn_se_summer_2cv, output_dict=True)
    level1_df = pd.DataFrame(level1_report).transpose()
    # level1_df.to_excel("secnnlevel1_2cv_classification_report.xlsx", index=True)
    level1_df.to_excel("secnnlevel1_2cv_" + str(i) + "classification_report.xlsx", index=True)

    print('a_cnn_se_summer_2000_AC=', a_cnn_se_summer_2cv_AC)
    fault0_2cv = accuracy_score(summer_2cv_test_Y[:144], a_cnn_se_summer_2cv[:144])
    fault1_2cv = accuracy_score(summer_2cv_test_Y[144:288], a_cnn_se_summer_2cv[144:288])
    fault2_2cv = accuracy_score(summer_2cv_test_Y[288:432], a_cnn_se_summer_2cv[288:432])
    fault3_2cv = accuracy_score(summer_2cv_test_Y[432:576], a_cnn_se_summer_2cv[432:576])
    fault4_2cv = accuracy_score(summer_2cv_test_Y[576:720], a_cnn_se_summer_2cv[576:720])
    fault5_2cv = accuracy_score(summer_2cv_test_Y[720:864], a_cnn_se_summer_2cv[720:864])
    fault6_2cv = accuracy_score(summer_2cv_test_Y[864:1008], a_cnn_se_summer_2cv[864:1008])
    fault7_2cv = accuracy_score(summer_2cv_test_Y[1008:1152], a_cnn_se_summer_2cv[1008:1152])
    fault8_2cv = accuracy_score(summer_2cv_test_Y[1152:1296], a_cnn_se_summer_2cv[1152:1296])
    fault9_2cv = accuracy_score(summer_2cv_test_Y[1296:1440], a_cnn_se_summer_2cv[1296:1440])
    fault10_2cv = accuracy_score(summer_2cv_test_Y[1440:1584], a_cnn_se_summer_2cv[1440:1584])
    fault11_2cv = accuracy_score(summer_2cv_test_Y[1584:1728], a_cnn_se_summer_2cv[1584:1728])
    fault12_2cv = accuracy_score(summer_2cv_test_Y[1728:1872], a_cnn_se_summer_2cv[1728:1872])

    f0.append(fault0_2cv)
    f1.append(fault1_2cv)
    f2.append(fault2_2cv)
    f3.append(fault3_2cv)
    f4.append(fault4_2cv)
    f5.append(fault5_2cv)
    f6.append(fault6_2cv)
    f7.append(fault7_2cv)
    f8.append(fault8_2cv)
    f9.append(fault9_2cv)
    f10.append(fault10_2cv)
    f11.append(fault11_2cv)
    f12.append(fault12_2cv)
    print('\n\n\n')
    print(acc)



for i in range(0,5):
    model3 = cnnse()
    cnn_se_summer_3cv =model3.fit(summer_3cv_train_X, summer_3cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
    det_cnn_se_summer_3cv = model3.predict(summer_3cv_test_X)
    a_cnn_se_summer_3cv = np.argmax(det_cnn_se_summer_3cv, axis=1)
    a_cnn_se_summer_3cv =  a_cnn_se_summer_3cv.reshape(-1, 1)
    # 输出总的的故障检测与分类精度
    a_cnn_se_summer_3cv_AC = accuracy_score(summer_3cv_test_Y, a_cnn_se_summer_3cv)
    acc.append(a_cnn_se_summer_3cv_AC)
    level1_cfm = confusion_matrix(summer_3cv_test_Y, a_cnn_se_summer_3cv)
    level1_conf = pd.DataFrame(level1_cfm).transpose()
    # level1_conf.to_excel("secnnlevel1_3cv_confusion_matrix.xlsx", index=True)
    level1_conf.to_excel("secnnlevel1_3cv_" + str(i) + "confusion_matrix.xlsx", index=True)
    print(level1_cfm)
    level1_report = classification_report(summer_3cv_test_Y, a_cnn_se_summer_3cv, output_dict=True)
    level1_df = pd.DataFrame(level1_report).transpose()
    # level1_df.to_excel("secnnlevel1_3cv_classification_report.xlsx", index=True)
    level1_df.to_excel("secnnlevel1_3cv_" + str(i) + "classification_report.xlsx", index=True)

    print('a_cnn_se_summer_2000_AC=', a_cnn_se_summer_3cv_AC)
    fault0_3cv = accuracy_score(summer_3cv_test_Y[:144], a_cnn_se_summer_3cv[:144])
    fault1_3cv = accuracy_score(summer_3cv_test_Y[144:288], a_cnn_se_summer_3cv[144:288])
    fault2_3cv = accuracy_score(summer_3cv_test_Y[288:432], a_cnn_se_summer_3cv[288:432])
    fault3_3cv = accuracy_score(summer_3cv_test_Y[432:576], a_cnn_se_summer_3cv[432:576])
    fault4_3cv = accuracy_score(summer_3cv_test_Y[576:720], a_cnn_se_summer_3cv[576:720])
    fault5_3cv = accuracy_score(summer_3cv_test_Y[720:864], a_cnn_se_summer_3cv[720:864])
    fault6_3cv = accuracy_score(summer_3cv_test_Y[864:1008], a_cnn_se_summer_3cv[864:1008])
    fault7_3cv = accuracy_score(summer_3cv_test_Y[1008:1152], a_cnn_se_summer_3cv[1008:1152])
    fault8_3cv = accuracy_score(summer_3cv_test_Y[1152:1296], a_cnn_se_summer_3cv[1152:1296])
    fault9_3cv = accuracy_score(summer_3cv_test_Y[1296:1440], a_cnn_se_summer_3cv[1296:1440])
    fault10_3cv = accuracy_score(summer_3cv_test_Y[1440:1584], a_cnn_se_summer_3cv[1440:1584])
    fault11_3cv = accuracy_score(summer_3cv_test_Y[1584:1728], a_cnn_se_summer_3cv[1584:1728])
    fault12_3cv = accuracy_score(summer_3cv_test_Y[1728:1872], a_cnn_se_summer_3cv[1728:1872])

    f0.append(fault0_3cv)
    f1.append(fault1_3cv)
    f2.append(fault2_3cv)
    f3.append(fault3_3cv)
    f4.append(fault4_3cv)
    f5.append(fault5_3cv)
    f6.append(fault6_3cv)
    f7.append(fault7_3cv)
    f8.append(fault8_3cv)
    f9.append(fault9_3cv)
    f10.append(fault10_3cv)
    f11.append(fault11_3cv)
    f12.append(fault12_3cv)
    print('\n\n\n')
    print(acc)

for i in range(0,5):
    model4 = cnnse()
    cnn_se_summer_4cv =model4.fit(summer_4cv_train_X, summer_4cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
    det_cnn_se_summer_4cv = model4.predict(summer_4cv_test_X)
    a_cnn_se_summer_4cv = np.argmax(det_cnn_se_summer_4cv, axis=1)
    a_cnn_se_summer_4cv =  a_cnn_se_summer_4cv.reshape(-1, 1)
    # 输出总的的故障检测与分类精度
    a_cnn_se_summer_4cv_AC = accuracy_score(summer_4cv_test_Y, a_cnn_se_summer_4cv)
    acc.append(a_cnn_se_summer_4cv_AC)
    level1_cfm = confusion_matrix(summer_4cv_test_Y, a_cnn_se_summer_4cv)
    level1_conf = pd.DataFrame(level1_cfm).transpose()
    # level1_conf.to_excel("level1_4cv_confusion_matrix.xlsx", index=True)
    level1_conf.to_excel("secnnlevel1_4cv_" + str(i) + "confusion_matrix.xlsx", index=True)
    print(level1_cfm)
    level1_report = classification_report(summer_4cv_test_Y, a_cnn_se_summer_4cv, output_dict=True)
    level1_df = pd.DataFrame(level1_report).transpose()
    # level1_df.to_excel("level1_4cv_classification_report.xlsx", index=True)
    level1_df.to_excel("secnnlevel1_4cv_" + str(i) + "classification_report.xlsx", index=True)

    print('a_cnn_se_summer_2000_AC=', a_cnn_se_summer_4cv_AC)
    fault0_4cv = accuracy_score(summer_4cv_test_Y[:144], a_cnn_se_summer_4cv[:144])
    fault1_4cv = accuracy_score(summer_4cv_test_Y[144:288], a_cnn_se_summer_4cv[144:288])
    fault2_4cv = accuracy_score(summer_4cv_test_Y[288:432], a_cnn_se_summer_4cv[288:432])
    fault3_4cv = accuracy_score(summer_4cv_test_Y[432:576], a_cnn_se_summer_4cv[432:576])
    fault4_4cv = accuracy_score(summer_4cv_test_Y[576:720], a_cnn_se_summer_4cv[576:720])
    fault5_4cv = accuracy_score(summer_4cv_test_Y[720:864], a_cnn_se_summer_4cv[720:864])
    fault6_4cv = accuracy_score(summer_4cv_test_Y[864:1008], a_cnn_se_summer_4cv[864:1008])
    fault7_4cv = accuracy_score(summer_4cv_test_Y[1008:1152], a_cnn_se_summer_4cv[1008:1152])
    fault8_4cv = accuracy_score(summer_4cv_test_Y[1152:1296], a_cnn_se_summer_4cv[1152:1296])
    fault9_4cv = accuracy_score(summer_4cv_test_Y[1296:1440], a_cnn_se_summer_4cv[1296:1440])
    fault10_4cv = accuracy_score(summer_4cv_test_Y[1440:1584], a_cnn_se_summer_4cv[1440:1584])
    fault11_4cv = accuracy_score(summer_4cv_test_Y[1584:1728], a_cnn_se_summer_4cv[1584:1728])
    fault12_4cv = accuracy_score(summer_4cv_test_Y[1728:1872], a_cnn_se_summer_4cv[1728:1872])

    f0.append(fault0_4cv)
    f1.append(fault1_4cv)
    f2.append(fault2_4cv)
    f3.append(fault3_4cv)
    f4.append(fault4_4cv)
    f5.append(fault5_4cv)
    f6.append(fault6_4cv)
    f7.append(fault7_4cv)
    f8.append(fault8_4cv)
    f9.append(fault9_4cv)
    f10.append(fault10_4cv)
    f11.append(fault11_4cv)
    f12.append(fault12_4cv)
    print('\n\n\n')
    print(acc)






for i in range(0,5):
    model5 = cnnse()
    cnn_se_summer_5cv =model5.fit(summer_5cv_train_X, summer_5cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
    det_cnn_se_summer_5cv = model5.predict(summer_5cv_test_X)
    a_cnn_se_summer_5cv = np.argmax(det_cnn_se_summer_5cv, axis=1)
    a_cnn_se_summer_5cv =  a_cnn_se_summer_5cv.reshape(-1, 1)
    # 输出总的的故障检测与分类精度
    a_cnn_se_summer_5cv_AC = accuracy_score(summer_5cv_test_Y, a_cnn_se_summer_5cv)
    acc.append(a_cnn_se_summer_5cv_AC)
    level1_cfm = confusion_matrix(summer_5cv_test_Y, a_cnn_se_summer_5cv)
    level1_conf = pd.DataFrame(level1_cfm).transpose()
    # level1_conf.to_excel("level1_5cv_confusion_matrix.xlsx", index=True)
    level1_conf.to_excel("secnnlevel1_5cv_" + str(i) + "confusion_matrix.xlsx", index=True)
    print(level1_cfm)
    level1_report = classification_report(summer_5cv_test_Y, a_cnn_se_summer_5cv, output_dict=True)
    level1_df = pd.DataFrame(level1_report).transpose()
    # level1_df.to_excel("level1_5cv_classification_report.xlsx", index=True)
    level1_df.to_excel("secnnlevel1_5cv_" + str(i) + "classification_report.xlsx", index=True)

    print('a_secnn_se_summer_2000_AC=', a_cnn_se_summer_5cv_AC)
    fault0_5cv = accuracy_score(summer_5cv_test_Y[:144], a_cnn_se_summer_5cv[:144])
    fault1_5cv = accuracy_score(summer_5cv_test_Y[144:288], a_cnn_se_summer_5cv[144:288])
    fault2_5cv = accuracy_score(summer_5cv_test_Y[288:432], a_cnn_se_summer_5cv[288:432])
    fault3_5cv = accuracy_score(summer_5cv_test_Y[432:576], a_cnn_se_summer_5cv[432:576])
    fault4_5cv = accuracy_score(summer_5cv_test_Y[576:720], a_cnn_se_summer_5cv[576:720])
    fault5_5cv = accuracy_score(summer_5cv_test_Y[720:864], a_cnn_se_summer_5cv[720:864])
    fault6_5cv = accuracy_score(summer_5cv_test_Y[864:1008], a_cnn_se_summer_5cv[864:1008])
    fault7_5cv = accuracy_score(summer_5cv_test_Y[1008:1152], a_cnn_se_summer_5cv[1008:1152])
    fault8_5cv = accuracy_score(summer_5cv_test_Y[1152:1296], a_cnn_se_summer_5cv[1152:1296])
    fault9_5cv = accuracy_score(summer_5cv_test_Y[1296:1440], a_cnn_se_summer_5cv[1296:1440])
    fault10_5cv = accuracy_score(summer_5cv_test_Y[1440:1584], a_cnn_se_summer_5cv[1440:1584])
    fault11_5cv = accuracy_score(summer_5cv_test_Y[1584:1728], a_cnn_se_summer_5cv[1584:1728])
    fault12_5cv = accuracy_score(summer_5cv_test_Y[1728:1872], a_cnn_se_summer_5cv[1728:1872])

    f0.append(fault0_5cv)
    f1.append(fault1_5cv)
    f2.append(fault2_5cv)
    f3.append(fault3_5cv)
    f4.append(fault4_5cv)
    f5.append(fault5_5cv)
    f6.append(fault6_5cv)
    f7.append(fault7_5cv)
    f8.append(fault8_5cv)
    f9.append(fault9_5cv)
    f10.append(fault10_5cv)
    f11.append(fault11_5cv)
    f12.append(fault12_5cv)
    print('\n\n\n')
    print(acc)

result_excel = pd.DataFrame()
result_excel["f0_acc"] = f0
result_excel["f1_acc"] = f1
result_excel["f2_acc"] = f2
result_excel["f3_acc"] = f3
result_excel["f4_acc"] = f4
result_excel["f5_acc"] = f5
result_excel["f6_acc"] = f6
result_excel["f7_acc"] = f7
result_excel["f8_acc"] = f8
result_excel["f9_acc"] = f9
result_excel["f10_acc"] = f10
result_excel["f11_acc"] = f11
result_excel["f12_acc"] = f12
# result_excel["acc"] = acc

result_excel.to_excel("secnn_level1_1cv_5cv_acc.xlsx")
