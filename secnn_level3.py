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

from chiller_data import lev3_1cv_train_X, lev3_1cv_train_Y,lev3_1cv_test_X, lev3_1cv_test_Y,lev3_2cv_train_X, lev3_2cv_train_Y,lev3_2cv_test_X, lev3_2cv_test_Y,lev3_3cv_train_X, lev3_3cv_train_Y,lev3_3cv_test_X, lev3_3cv_test_Y,lev3_4cv_train_X, lev3_4cv_train_Y,lev3_4cv_test_X, lev3_4cv_test_Y,lev3_5cv_train_X, lev3_5cv_train_Y,lev3_5cv_test_X, lev3_5cv_test_Y
# Nada = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

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

    cnn_se_lev3_2000 = tf.keras.models.Model(inputs=input_deep,outputs=[output])
    cnn_se_lev3_2000_det1 = cnn_se_lev3_2000
    cnn_se_lev3_2000_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])

    return cnn_se_lev3_2000_det1
# cnn_se_lev3_2000_det1.summary()

time1 = time.time()

###一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合keras.callbacks.EarlyStopping就是用来提前结束训练的。
callback1 = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=patienc, restore_best_weights=True)]

model1 = cnnse()
cnn_se_lev3_1cv =model1.fit(lev3_1cv_train_X, lev3_1cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
det_cnn_se_lev3_1cv = model1.predict(lev3_1cv_test_X)
a_cnn_se_lev3_1cv = np.argmax(det_cnn_se_lev3_1cv, axis=1)
a_cnn_se_lev3_1cv =  a_cnn_se_lev3_1cv.reshape(-1, 1)
# 输出总的的故障检测与分类精度
a_cnn_se_lev3_1cv_AC = accuracy_score(lev3_1cv_test_Y, a_cnn_se_lev3_1cv)
acc.append(a_cnn_se_lev3_1cv_AC)
level3_cfm = confusion_matrix(lev3_1cv_test_Y, a_cnn_se_lev3_1cv)
level3_conf = pd.DataFrame(level3_cfm).transpose()
level3_conf.to_excel("level3_1cv_confusion_matrix.xlsx", index=True)
print(level3_cfm)
level3_report = classification_report(lev3_1cv_test_Y, a_cnn_se_lev3_1cv, output_dict=True)
level3_df = pd.DataFrame(level3_report).transpose()
level3_df.to_excel("level3_1cv_classification_report.xlsx", index=True)

print('a_cnn_se_lev3_1cv_AC=', a_cnn_se_lev3_1cv_AC)
print('\n\n\n')
#
#
#
#
#
model2 = cnnse()
cnn_se_lev3_2cv =model2.fit(lev3_2cv_train_X, lev3_2cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
det_cnn_se_lev3_2cv = model2.predict(lev3_2cv_test_X)
a_cnn_se_lev3_2cv = np.argmax(det_cnn_se_lev3_2cv, axis=1)
a_cnn_se_lev3_2cv =  a_cnn_se_lev3_2cv.reshape(-1, 1)
# 输出总的的故障检测与分类精度
a_cnn_se_lev3_2cv_AC = accuracy_score(lev3_2cv_test_Y, a_cnn_se_lev3_2cv)
acc.append(a_cnn_se_lev3_2cv_AC)
level3_cfm = confusion_matrix(lev3_2cv_test_Y, a_cnn_se_lev3_2cv)
level3_conf = pd.DataFrame(level3_cfm).transpose()
level3_conf.to_excel("level3_2cv_confusion_matrix.xlsx", index=True)
print(level3_cfm)
level3_report = classification_report(lev3_2cv_test_Y, a_cnn_se_lev3_2cv, output_dict=True)
level3_df = pd.DataFrame(level3_report).transpose()
level3_df.to_excel("level3_2cv_classification_report.xlsx", index=True)

print('a_cnn_se_lev3_2cv_AC =', a_cnn_se_lev3_2cv_AC)
print('\n\n\n')



model3 = cnnse()
cnn_se_lev3_3cv =model3.fit(lev3_3cv_train_X, lev3_3cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
det_cnn_se_lev3_3cv = model3.predict(lev3_3cv_test_X)
a_cnn_se_lev3_3cv = np.argmax(det_cnn_se_lev3_3cv, axis=1)
a_cnn_se_lev3_3cv =  a_cnn_se_lev3_3cv.reshape(-1, 1)
# 输出总的的故障检测与分类精度
a_cnn_se_lev3_3cv_AC = accuracy_score(lev3_3cv_test_Y, a_cnn_se_lev3_3cv)
acc.append(a_cnn_se_lev3_3cv_AC)
level3_cfm = confusion_matrix(lev3_3cv_test_Y, a_cnn_se_lev3_3cv)
level3_conf = pd.DataFrame(level3_cfm).transpose()
level3_conf.to_excel("level3_3cv_confusion_matrix.xlsx", index=True)
print(level3_cfm)
level3_report = classification_report(lev3_3cv_test_Y, a_cnn_se_lev3_3cv, output_dict=True)
level3_df = pd.DataFrame(level3_report).transpose()
level3_df.to_excel("level3_3cv_classification_report.xlsx", index=True)

print('a_cnn_se_lev3_3cv_AC=', a_cnn_se_lev3_3cv_AC)
print('\n\n\n')






model4 = cnnse()
cnn_se_lev3_4cv =model4.fit(lev3_4cv_train_X, lev3_4cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
det_cnn_se_lev3_4cv = model4.predict(lev3_4cv_test_X)
a_cnn_se_lev3_4cv = np.argmax(det_cnn_se_lev3_4cv, axis=1)
a_cnn_se_lev3_4cv =  a_cnn_se_lev3_4cv.reshape(-1, 1)
# 输出总的的故障检测与分类精度
a_cnn_se_lev3_4cv_AC = accuracy_score(lev3_4cv_test_Y, a_cnn_se_lev3_4cv)
acc.append(a_cnn_se_lev3_4cv_AC)
level3_cfm = confusion_matrix(lev3_4cv_test_Y, a_cnn_se_lev3_4cv)
level3_conf = pd.DataFrame(level3_cfm).transpose()
level3_conf.to_excel("level3_4cv_confusion_matrix.xlsx", index=True)
print(level3_cfm)
level3_report = classification_report(lev3_4cv_test_Y, a_cnn_se_lev3_4cv, output_dict=True)
level3_df = pd.DataFrame(level3_report).transpose()
level3_df.to_excel("level3_4cv_classification_report.xlsx", index=True)

print('a_cnn_se_lev3_4cv_AC=', a_cnn_se_lev3_4cv_AC)
print('\n\n\n')






model5 = cnnse()
cnn_se_lev3_5cv =model5.fit(lev3_5cv_train_X, lev3_5cv_train_Y,callbacks=callback1, batch_size=100, epochs=1000, verbose=2)
det_cnn_se_lev3_5cv = model5.predict(lev3_5cv_test_X)
a_cnn_se_lev3_5cv = np.argmax(det_cnn_se_lev3_5cv, axis=1)
a_cnn_se_lev3_5cv =  a_cnn_se_lev3_5cv.reshape(-1, 1)
# 输出总的的故障检测与分类精度
a_cnn_se_lev3_5cv_AC = accuracy_score(lev3_5cv_test_Y, a_cnn_se_lev3_5cv)
acc.append(a_cnn_se_lev3_5cv_AC)
level3_cfm = confusion_matrix(lev3_5cv_test_Y, a_cnn_se_lev3_5cv)
level3_conf = pd.DataFrame(level3_cfm).transpose()
level3_conf.to_excel("level3_5cv_confusion_matrix.xlsx", index=True)
print(level3_cfm)
level3_report = classification_report(lev3_5cv_test_Y, a_cnn_se_lev3_5cv, output_dict=True)
level3_df = pd.DataFrame(level3_report).transpose()
level3_df.to_excel("level3_5cv_classification_report.xlsx", index=True)

print('a_cnn_se_lev3_5cv_AC=', a_cnn_se_lev3_5cv_AC)
print('\n\n\n')

result_excel = pd.DataFrame()
result_excel["secnn_level3_1cv_5cv_acc"] = acc
result_excel.to_excel("secnn_level3_1cv_5cv_acc.xlsx")
