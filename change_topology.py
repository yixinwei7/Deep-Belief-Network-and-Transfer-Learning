##

from DBN import DBN
from nn_tf import NN
import numpy as np
import  csv
from sklearn import metrics                           #导入验证包
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from opts import DLOption


def autoNorm(data):
    mins = data.min(0)
    maxs = data.max(0)
    ranges = maxs - mins
    normData = np.zeros(np.shape(data))
    row = data.shape[0]
    normData = data - np.tile(mins, (row, 1))
    normData = normData / np.tile(ranges, (row, 1))
    return normData


with open('E:/数据集/2020年3_6月nyiso数据/all/x_3_6month_all.csv','r', encoding="utf-8") as file:                        #（有差异的 IEEE14的 训练数据）   {即分布不同的数据}
#with open('E:/数据集/2020年3_6月nyiso数据/all_有名值_数据/x_3_6month_all.csv','r', encoding="utf-8") as file:                        #（有差异的 IEEE14的 训练数据）   {即分布不同的数据}
#with open('E:/数据集/源域/6_1_0.3939/6_1_all.csv','r', encoding="utf-8") as file:
#with open('E:/数据集/2020年3_6月加入不同攻击个数和变噪声的数据18_2/all/change_n_zaosheng_case14_3_6month_all.csv','r', encoding="utf-8") as file:
    reader = csv.reader(file)
    a = []
    for item in reader:
        a.append(item)
    a =[[float(x) for x in item] for item in a]      #将矩阵数据转化为浮点型
    data = np.array(a)
    x_data = autoNorm(data[:,0:54])
    y_data = data[:,[54,55]]
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, test_size=0.8, random_state=None,shuffle=True)

#with open('E:/数据集/2020年3_6月标准IEEE14_nyiso数据/all/normal_case14_3_6month_all.csv','r', encoding="utf-8") as file:                 #(标准IEEE14用于DBN无监督预训练数据)
with open('E:/数据集/不同比例的真实样本/100：1/3_6month/normal_case14_3_6month_all.csv','r', encoding="utf-8") as file:
#with open('E:/数据集/2020年3_6月标准IEEE14_nyiso数据/all_有名值_数据/normal_case14_3_6month_all.csv','r', encoding="utf-8") as file:                 #(标准IEEE14用于DBN无监督预训练数据)
    reader = csv.reader(file)
    a = []
    for item in reader:
        a.append(item)
    a =[[float(x) for x in item] for item in a]      #将矩阵数据转化为浮点型
    data = np.array(a)
    x_data_pretrain = autoNorm(data[:,0:54])
    y_data_pretrain = data[:,[54,55]]
    x_data_pretrain = x_data_pretrain.astype(np.float32)
    y_data_pretrain = y_data_pretrain.astype(np.float32)


#with open('E:/数据集/2020年2月nyiso数据集/2月1日2日数据/all/x_2month12_all.csv','r', encoding="utf-8") as file:              #(标准IEEE14的测试数据)
with open('E:/数据集/2020年2月nyiso数据集/2月1日数据/all/x_2month1_all.csv', 'r',encoding="utf-8") as file:         # (标准IEEE14的测试数据)
#with open('E:/数据集/2020年2月nyiso数据集/2月1日数据/all_有名值_数据/x_2month1_all.csv', 'r', encoding="utf-8") as file:  # (标准IEEE14的测试数据)
    reader = csv.reader(file)
    a = []
    for item in reader:
        a.append(item)
    a =[[float(x) for x in item] for item in a]      #将矩阵数据转化为浮点型
    data = np.array(a)
    x_data_migration = autoNorm(data[:,0:54])
    y_data_migration = data[:,[54,55]]
    x_data_migration = x_data_migration.astype(np.float32)
    y_data_migration = y_data_migration.astype(np.float32)
    x_migration_train, x_migration_test, y_migration_train, y_migration_test = train_test_split(x_data_migration, y_data_migration, test_size=0.5, random_state=None,shuffle=True)

with open('E:/数据集/2020年2月nyiso数据集/2月2日数据/x_2month2_all.csv', 'r',encoding="utf-8") as file:         # (标准IEEE14的测试数据)
#with open('E:/数据集/2020年2月nyiso数据集/2月1日数据/all_有名值_数据/x_2month1_all.csv', 'r', encoding="utf-8") as file:  # (标准IEEE14的测试数据)
    reader = csv.reader(file)
    a = []
    for item in reader:
        a.append(item)
    a =[[float(x) for x in item] for item in a]      #将矩阵数据转化为浮点型
    data = np.array(a)
    x_data_migration_test = autoNorm(data[:,0:54])
    y_data_migration_test = data[:,[54,55]]
    x_data_migration_test = x_data_migration_test.astype(np.float32)
    y_data_migration_test = y_data_migration_test.astype(np.float32)


opts = DLOption(18,450, 0.1, 0.2,1000, 0, 0., 0.,0.01,7000,300,0.001,50000)
dbn = DBN([40,20,12], opts, x_data_pretrain)
dbn.train()


nn = NN([40,20,12], [10,6],opts, x_data_train, y_data_train,x_data_test, y_data_test,x_migration_train,y_migration_train,x_migration_test,y_migration_test,x_data_migration_test,y_data_migration_test,[10])
nn.load_from_dbn(dbn)
nn.train()
#print( np.mean(np.argmax(y_data_test, axis=1) == nn.predict(x_data_test)))


nn.train_migration()
nn.train_migration_all()




