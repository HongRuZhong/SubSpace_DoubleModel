import numpy as np
import pandas as pd
from SOM_SubSpace_DoubleModel.SOM_Subspace_Model import SSM
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

file_Path="D:\\Data\\SOM聚类_子空间分类_双层模型\\岩性数据\\T6_Data\\"
out_path="D:\\Data\\SOM聚类_子空间分类_双层模型\\岩性数据\\T6_Data\\Out_Result\\"
Data_Path=file_Path+"Data_T6.txt"
SOM_Params_Path="D:\\Data\\SOM聚类_子空间分类_双层模型\\岩性数据\\各类数据SOM结果输出\\V3_BlobsSOM处理\\20x20神经元\\各个神经元的权值及类别标签.txt"

def MakeConfusionMatrixWithACC(conf_mat,label_name):
	ny = conf_mat.shape[0]
	nx = conf_mat.shape[1]
	#创建pd格式混淆矩阵
	dst=pd.DataFrame(conf_mat,index=label_name,columns=label_name,dtype=float)
	#计算召回率
	recall=[conf_mat[x,x]/np.sum(conf_mat[x,:]) for x in range(ny)]
	#计算准确率
	acc=[conf_mat[x,x]/np.sum(conf_mat[:,x]) for x in range(nx)]
	#在准确率上添加总体准确率
	trace_sum=np.sum(np.trace(conf_mat))#迹的和
	conf_sum=np.sum(conf_mat)
	acc.append(trace_sum/conf_sum)
	#在pd上添加召回率列
	dst["Recall"]=recall
	#在pd上添加行--准确率
	dst.loc["Accuracy"]=acc
	return dst

def main():
	All_Data = np.loadtxt(Data_Path, skiprows=1, delimiter='\t')
	train_attri=All_Data[:,1:7]
	# print(train_attri[0:5])
	train_true_label=All_Data[:,7]
	# print(train_true_label[0:5])
	train_SubSpace_label=All_Data[:,8]
	# print(train_SubSpace_label[0:5])

	# train_attri=StandardScaler().fit_transform(train_attri)
	Use_Model=xgb.sklearn.XGBClassifier()
	# Use_Model=GradientBoostingClassifier()
	# Use_Model=DecisionTreeClassifier()
	# Use_Model=MLPClassifier((64,))
	# Use_Model=GaussianNB()
	# Use_Model=lgb.sklearn.LGBMClassifier()
	filename="T6_贝叶斯_T400_"
	ssm=SSM(Use_Model)
	ssm.fit(train_attri,train_true_label,train_SubSpace_label)

	# Test_Paras = np.loadtxt(SOM_Params_Path, skiprows=0, delimiter='\t')
	# print(Test_Paras)
	# test_params=Test_Paras[:,0:3]
	# print(test_params)
	pred_label=ssm.predict_withType(train_attri,train_SubSpace_label)
	print("ACC",accuracy_score(train_true_label,pred_label))
	#输出结果
	np.savetxt(out_path + filename+"预测结果.txt", pred_label, fmt='%d', delimiter='\t',
			   comments='')
	#输出混淆矩阵
	test_confu = confusion_matrix(train_true_label, pred_label)
	confu_label = np.linspace(1, 6, 6)
	out_confu=MakeConfusionMatrixWithACC(test_confu,confu_label)
	np.savetxt(out_path + filename+"混淆矩阵.txt", out_confu, fmt='%.02f', delimiter='\t',
			   comments='')
	# #输出子空间索引
	# Test_SubSpace_Index=ssm.get_Test_SubSpace_Index()
	# Test_SubSpace_Index=np.vstack(Test_SubSpace_Index)
	# Test_SubSpace_Index=Test_SubSpace_Index.reshape(-1)
	# headline="SubSpace_Index"
	# np.savetxt(out_path+filename+"SubSpace_Index.txt", Test_SubSpace_Index, fmt='%d', delimiter='\t', header=headline, comments='')
	#输出子空间类别个数
	SubSpace_Number=ssm.get_conut_Each_SubSpace()
	np.savetxt(out_path + filename+"SubSpace_Number.txt", SubSpace_Number, fmt='%d', delimiter='\t',
			   comments='')
	#输出子空间概率
	SubSpace_Number = ssm.get_Pro_Each_SubSpace()
	np.savetxt(out_path + filename+"SubSpace_Pro.txt", SubSpace_Number, fmt='%.04f', delimiter='\t',
			   comments='')

if __name__=='__main__':
	main()

