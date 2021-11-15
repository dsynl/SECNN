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

from chiller_data import lev1_1cv_train_X, lev1_1cv_train_Y,lev1_1cv_test_X, lev1_1cv_test_Y,lev1_2cv_train_X, lev1_2cv_train_Y,lev1_2cv_test_X, lev1_2cv_test_Y,lev1_3cv_train_X, lev1_3cv_train_Y,lev1_3cv_test_X, lev1_3cv_test_Y,lev1_4cv_train_X, lev1_4cv_train_Y,lev1_4cv_test_X, lev1_4cv_test_Y,lev1_5cv_train_X, lev1_5cv_train_Y,lev1_5cv_test_X, lev1_5cv_test_Y

min_max_scaler = MinMaxScaler()
# np.random.seed(6)
np.random.seed(6)
o = 1
p = 1
patienc=30
acc = []

def cnnse():

    input_deep = tf.keras.layers.Input(shape=(8, 1))
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
    output = tf.keras.layers.Dense(8, activation='softmax')(dp)

    cnn_se_lev1_2000 = tf.keras.models.Model(inputs=input_deep,outputs=[output])
    cnn_se_lev1_2000_det1 = cnn_se_lev1_2000
    cnn_se_lev1_2000_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])

    return cnn_se_lev1_2000_det1
# cnn_se_lev1_2000_det1.summary()

time1 = time.time()

###一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合keras.callbacks.EarlyStopping就是用来提前结束训练的。
callback1 = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=patienc, restore_best_weights=True)]
for i in range(0,10):
    model1 = cnnse()
    cnn_se_lev1_1cv =model1.fit(lev1_1cv_train_X, lev1_1cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
    det_cnn_se_lev1_1cv = model1.predict(lev1_1cv_test_X)
    a_cnn_se_lev1_1cv = np.argmax(det_cnn_se_lev1_1cv, axis=1)
    a_cnn_se_lev1_1cv =  a_cnn_se_lev1_1cv.reshape(-1, 1)
    # 输出总的的故障检测与分类精度
    a_cnn_se_lev1_1cv_AC = accuracy_score(lev1_1cv_test_Y, a_cnn_se_lev1_1cv)
    acc.append(a_cnn_se_lev1_1cv_AC)
    level1_cfm = confusion_matrix(lev1_1cv_test_Y, a_cnn_se_lev1_1cv)
    level1_conf = pd.DataFrame(level1_cfm).transpose()
    level1_conf.to_excel("level1_1cv_"+str(i)+"confusion_matrix.xlsx", index=True)
    print(level1_cfm)
    level1_report = classification_report(lev1_1cv_test_Y, a_cnn_se_lev1_1cv, output_dict=True)
    level1_df = pd.DataFrame(level1_report).transpose()
    level1_df.to_excel("level1_1cv_"+str(i)+"classification_report.xlsx", index=True)

    print('a_cnn_se_lev1_1cv_AC=', a_cnn_se_lev1_1cv_AC)
    print('\n\n\n')
    print(acc)

#
#
#
for i in range(0,10):
    model2 = cnnse()
    cnn_se_lev1_2cv =model2.fit(lev1_2cv_train_X, lev1_2cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
    det_cnn_se_lev1_2cv = model2.predict(lev1_2cv_test_X)
    a_cnn_se_lev1_2cv = np.argmax(det_cnn_se_lev1_2cv, axis=1)
    a_cnn_se_lev1_2cv =  a_cnn_se_lev1_2cv.reshape(-1, 1)
    # 输出总的的故障检测与分类精度
    a_cnn_se_lev1_2cv_AC = accuracy_score(lev1_2cv_test_Y, a_cnn_se_lev1_2cv)
    acc.append(a_cnn_se_lev1_2cv_AC)
    level1_cfm = confusion_matrix(lev1_2cv_test_Y, a_cnn_se_lev1_2cv)
    level1_conf = pd.DataFrame(level1_cfm).transpose()
    # level1_conf.to_excel("level1_2cv_confusion_matrix.xlsx", index=True)
    level1_conf.to_excel("level1_2cv_" + str(i) + "confusion_matrix.xlsx", index=True)
    print(level1_cfm)
    level1_report = classification_report(lev1_2cv_test_Y, a_cnn_se_lev1_2cv, output_dict=True)
    level1_df = pd.DataFrame(level1_report).transpose()
    # level1_df.to_excel("level1_2cv_classification_report.xlsx", index=True)
    level1_df.to_excel("level1_2cv_" + str(i) + "classification_report.xlsx", index=True)

    print('a_cnn_se_lev1_2cv_AC=', a_cnn_se_lev1_2cv_AC)
    print('\n\n\n')
    print(acc)



model3 = cnnse()
cnn_se_lev1_3cv =model3.fit(lev1_3cv_train_X, lev1_3cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
det_cnn_se_lev1_3cv = model3.predict(lev1_3cv_test_X)
a_cnn_se_lev1_3cv = np.argmax(det_cnn_se_lev1_3cv, axis=1)
a_cnn_se_lev1_3cv =  a_cnn_se_lev1_3cv.reshape(-1, 1)
# 输出总的的故障检测与分类精度
a_cnn_se_lev1_3cv_AC = accuracy_score(lev1_3cv_test_Y, a_cnn_se_lev1_3cv)
acc.append(a_cnn_se_lev1_3cv_AC)
level1_cfm = confusion_matrix(lev1_3cv_test_Y, a_cnn_se_lev1_3cv)
level1_conf = pd.DataFrame(level1_cfm).transpose()
level1_conf.to_excel("level1_3cv_confusion_matrix.xlsx", index=True)
print(level1_cfm)
level1_report = classification_report(lev1_3cv_test_Y, a_cnn_se_lev1_3cv, output_dict=True)
level1_df = pd.DataFrame(level1_report).transpose()
level1_df.to_excel("level1_3cv_classification_report.xlsx", index=True)

print('a_cnn_se_lev1_3cv_AC=', a_cnn_se_lev1_3cv_AC)
print('\n\n\n')






model4 = cnnse()
cnn_se_lev1_4cv =model4.fit(lev1_4cv_train_X, lev1_4cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
det_cnn_se_lev1_4cv = model4.predict(lev1_4cv_test_X)
a_cnn_se_lev1_4cv = np.argmax(det_cnn_se_lev1_4cv, axis=1)
a_cnn_se_lev1_4cv =  a_cnn_se_lev1_4cv.reshape(-1, 1)
# 输出总的的故障检测与分类精度
a_cnn_se_lev1_4cv_AC = accuracy_score(lev1_4cv_test_Y, a_cnn_se_lev1_4cv)
acc.append(a_cnn_se_lev1_4cv_AC)
level1_cfm = confusion_matrix(lev1_4cv_test_Y, a_cnn_se_lev1_4cv)
level1_conf = pd.DataFrame(level1_cfm).transpose()
level1_conf.to_excel("level1_4cv_confusion_matrix.xlsx", index=True)
print(level1_cfm)
level1_report = classification_report(lev1_4cv_test_Y, a_cnn_se_lev1_4cv, output_dict=True)
level1_df = pd.DataFrame(level1_report).transpose()
level1_df.to_excel("level1_4cv_classification_report.xlsx", index=True)

print('a_cnn_se_lev1_4cv_AC=', a_cnn_se_lev1_4cv_AC)
print('\n\n\n')






model5 = cnnse()
cnn_se_lev1_5cv =model5.fit(lev1_5cv_train_X, lev1_5cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
det_cnn_se_lev1_5cv = model5.predict(lev1_5cv_test_X)
a_cnn_se_lev1_5cv = np.argmax(det_cnn_se_lev1_5cv, axis=1)
a_cnn_se_lev1_5cv =  a_cnn_se_lev1_5cv.reshape(-1, 1)
# 输出总的的故障检测与分类精度
a_cnn_se_lev1_5cv_AC = accuracy_score(lev1_5cv_test_Y, a_cnn_se_lev1_5cv)
acc.append(a_cnn_se_lev1_5cv_AC)
level1_cfm = confusion_matrix(lev1_5cv_test_Y, a_cnn_se_lev1_5cv)
level1_conf = pd.DataFrame(level1_cfm).transpose()
level1_conf.to_excel("level1_5cv_confusion_matrix.xlsx", index=True)
print(level1_cfm)
level1_report = classification_report(lev1_5cv_test_Y, a_cnn_se_lev1_5cv, output_dict=True)
level1_df = pd.DataFrame(level1_report).transpose()
level1_df.to_excel("level1_5cv_classification_report.xlsx", index=True)

print('a_cnn_se_lev1_5cv_AC=', a_cnn_se_lev1_5cv_AC)
print('\n\n\n')

result_excel = pd.DataFrame()
result_excel["secnn_level1_1cv_5cv_acc"] = acc
result_excel.to_excel("secnn_level1_1cv_5cv_acc.xlsx")
