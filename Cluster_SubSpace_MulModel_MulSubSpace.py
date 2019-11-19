import numpy as np
import pandas as pd
from SOM_SubSpace_DoubleModel.SOM_Subspace_Model import SSM
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from Machine_Learning_Algoithm_Base_Sklearn_Dictionary import GetModules
from 使用决策树得到样本子空间 import Gei_Model_SubSpace
import copy
import time

file_Path="D:\\sudty\\图片聚类\\MNIST_聚类\\"
# file_Path="D:\\Data\\SOM聚类_子空间分类_双层模型\\岩性数据\\T6_Data\\"
# file_Path="D:\\Data\\SOM聚类_子空间分类_双层模型\\岩性数据\\Pri\\"
# file_Path="D:\\Data\\Data_机器学习常用数据集\\"
# file_Path="D:\\Data\\SOM聚类_子空间分类_双层模型\\V3\\决策树\\"
# file_Path="D:\\Data\\SOM聚类_子空间分类_双层模型\\岩性数据\\决策树划分子空间\\"
# out_path="D:\\Data\\SOM聚类_子空间分类_双层模型\\岩性数据\\决策树划分子空间\\Test_Result\\"
filename="MNIST_train"
out_path=file_Path+"MNIST_子空间结果\\MNIST_SOM_预测\\"

test_data_path="D:\\Data\\SOM聚类_子空间分类_双层模型\\岩性数据\\T6_Test\\Data_All_Well.txt"
Data_Path=file_Path+filename+".txt"
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

def Run(train_attri,train_true_label,train_SubSpace_label,test_attri,test_label,model,SubSpace_Number=None):
	# filename="T6_贝叶斯_T400_"
	ssm=SSM(model)
	ssm.fit(train_attri,train_true_label,train_SubSpace_label)
	pred_label=ssm.predict_Bp_DivideSubSpace(test_attri)
	# pred_label=ssm.predict_withType(test_attri,train_SubSpace_label)
	# pred_label=ssm.predict_withType(train_attri,train_SubSpace_label)
	# print("ACC",accuracy_score(train_true_label,pred_label))
	# #输出结果
	# np.savetxt(filename+"预测结果.txt", pred_label, fmt='%d', delimiter='\t',
	# 		   comments='')
	# #输出混淆矩阵
	# test_confu = confusion_matrix(test_label, pred_label)
	# confu_label = np.linspace(1, 9, 9)
	# out_confu=MakeConfusionMatrixWithACC(test_confu,confu_label)
	# np.savetxt(filename+"混淆矩阵.txt", out_confu, fmt='%.02f', delimiter='\t',
	# 		   comments='')
	# # #输出子空间索引
	# # Test_SubSpace_Index=ssm.get_Test_SubSpace_Index()
	# # Test_SubSpace_Index=np.vstack(Test_SubSpace_Index)
	# # Test_SubSpace_Index=Test_SubSpace_Index.reshape(-1)
	# # headline="SubSpace_Index"
	# # np.savetxt(out_path+filename+"SubSpace_Index.txt", Test_SubSpace_Index, fmt='%d', delimiter='\t', header=headline, comments='')
	# #输出子空间类别个数
	# SubSpace_Number=ssm.get_conut_Each_SubSpace()
	# np.savetxt(filename+"SubSpace_Number.txt", SubSpace_Number, fmt='%d', delimiter='\t',
	# 		   comments='')
	# #输出子空间概率
	# SubSpace_Number = ssm.get_Pro_Each_SubSpace()
	# np.savetxt(filename+"SubSpace_Pro.txt", SubSpace_Number, fmt='%.04f', delimiter='\t',
	# 		   comments='')
	return accuracy_score(test_label, pred_label),pred_label

#获得每个子空间的熵值
def Get_Entropy():
	All_Data = np.loadtxt(Data_Path, skiprows=1, delimiter='\t')
	train_attri = All_Data[:, 0:6]
	train_true_label = All_Data[:, 6]
	# SubSpace_Dic={7:"400",8:"225",9:"100",10:"49"}
	SubSpace_Dic = {12: "400", 13: "225", 14: "100", 15: "49"}
	model=""
	ssm = SSM(model)
	headline="No\tALL_Entropy\tAVE_Entropy"
	for x in SubSpace_Dic:
		train_SubSpace_label = All_Data[:, x]
		ALL_Entropy, AVE_Entropy = ssm.Get_Entropy_Each_SubSpace(train_attri, train_true_label, train_SubSpace_label)
		No=np.arange(0,int(SubSpace_Dic[x]),1)
		stack_entropy=np.column_stack((No,ALL_Entropy,AVE_Entropy))
		np.savetxt(out_path+"Entropy_SOM_SubSpace_"+SubSpace_Dic[x]+".txt",stack_entropy,fmt='%.06f',delimiter='\t',
				   header=headline,comments='')
#获得每个子空间的精度
def Get_ACC_Each_SubSpace(filepath):
	# print(filepath)
	All_Data = np.loadtxt(filepath+filename+".txt", skiprows=1, delimiter='\t')
	train_attri = All_Data[:, 0:784]
	# print(train_attri[0:5])
	train_true_label = All_Data[:, 784]
	# print(train_true_label)
	SubSpace_Dic = {785: "49", 786: "100", 787: "225", 788: "400",789:"625"}
	# SubSpace_Dic = {4: "400", 5: "225", 6: "100", 7: "49"}
	# SubSpace_Dic={7:"400",8:"225",9:"100",10:"49"}
	# SubSpace_Dic = {7: "50", 8: "100", 9: "150", 10: "200", 11: '250', 12: "300", 13: "350", 14: "400"}

	# SubSpace_Dic = {7: "1450",8:"1600" ,9: "1225", 10: "900", 11: "625",12:"400",13:"225",14:"100",15:"49"}
	# SubSpace_Dic = {12: "400", 13: "225", 14: "100", 15: "49"}
	models = GetModules()
	headline = "No\tACC"
	train_attri = StandardScaler().fit_transform(train_attri)
	# 非树模型 值训练SVM和贝叶斯
	for mi in range(2, 5):
		model_item = models[mi].items()
		model_name = ""
		Use_model = ""
		for key, values in model_item:
			model_name = key
			Use_model = values
		# print(model_name)
		# continue
		for x in SubSpace_Dic:
			# 子空间预测
			train_SubSpace_label = All_Data[:, x]  # 亚楠的数据是从1开始，所以要减1
			ssm=SSM(Use_model)
			ssm.fit(train_attri,train_true_label,train_SubSpace_label)
			SubSpace_ACC=ssm.get_SubSpace_ACC()
			No = np.arange(0, int(SubSpace_Dic[x]), 1)
			stack_acc = np.column_stack((No, SubSpace_ACC))
			np.savetxt(out_path + "SubSpace_ACC_"+model_name+"_SOM_" + SubSpace_Dic[x] + ".txt", stack_acc, fmt='%.04f',
					   delimiter='\t',
					   header=headline, comments='')
#使用决策树划分子空间，划分训练测试集，进行训练和预测
def DT_SubSpace_Run():
	All_Data = np.loadtxt(Data_Path, skiprows=1, delimiter='\t')
	train_attri = All_Data[:, 0:6]
	# print(train_attri[0:5])
	train_true_label = All_Data[:, 6]
	# print(train_true_label[0:5])
	# test_attri=train_attri
	# test_label=train_true_label
	train_attri,test_attri,train_true_label,test_label=train_test_split(train_attri,train_true_label,test_size=0.7,random_state=0)
	#529岩性数据
	# All_Data = np.loadtxt(file_Path+"岩性数据_Train.txt", skiprows=1, delimiter='\t')
	# train_attri = All_Data[:, 0:-1]
	# # print(train_attri[0:5])
	# train_true_label = All_Data[:, -1]
	# test_data=np.loadtxt(file_Path+"岩性数据_Test.txt", skiprows=1, delimiter='\t')
	# test_attri=test_data[:,0:-1]
	# test_label=test_data[:,-1]
	#8w岩性数据
	# All_Data = np.loadtxt(file_Path+"Pri_岩性数据训练集.txt", skiprows=1, delimiter='\t')
	# train_attri = All_Data[:, 0:-1]
	# # print(train_attri[0:5])
	# train_true_label = All_Data[:, -1]
	# test_data=np.loadtxt(file_Path+"Pri_岩性数据测试集.txt", skiprows=1, delimiter='\t')

	SubSpace_Number_list=[49,100,225,400]
	ACC_Time_List=[]
	All_Data_Predict_List=[]
	SubSpace_Predict_List=[]
	headline=''
	# for x in SubSpace_Number_list:
	for x in range(10,400,30):
		train_SubSpace_label,DT_Model,LeafToSubSpaceIndex=Gei_Model_SubSpace(train_attri,train_true_label,x)
		# print(train_SubSpace_label)
		ACC_Time,All_Data_Predict,SubSpace_Predict,hd=Run_Model(train_attri,
															 train_true_label,
															train_SubSpace_label,
															test_attri,
															test_label,
															DT_Model,
															LeafToSubSpaceIndex)
		headline+=hd
		ACC_Time_List.append(ACC_Time)
		All_Data_Predict_List.append(All_Data_Predict)
		SubSpace_Predict_List.append(SubSpace_Predict)
	All_Data_ALL_Result = np.column_stack(All_Data_Predict_List)
	np.savetxt(out_path + filename + "_All_Data_All_Result.txt", All_Data_ALL_Result, delimiter='\t', comments="",
			   header=headline, fmt='%d')
	SubSpace_All_Result = np.column_stack(SubSpace_Predict_List)
	np.savetxt(out_path + filename + "_SubSpace_All_Result.txt", SubSpace_All_Result, delimiter='\t', comments="",
			   header=headline, fmt='%d')
	ACC_Time_file = np.row_stack(ACC_Time_List)
	headline = headline.split('\t')[0:-1]
	ACC_Time_file = pd.DataFrame(ACC_Time_file, index=headline,
								 columns=['All_Time', 'ALL_ACC', 'SubS_Time', 'SubS_ACC'])
	ACC_Time_file.sort_index(axis=0)
	ACC_Time_file.to_csv(out_path + filename + "_ACC_Time_Pd.txt", '\t', index=True, header=True)  # 文件不能有汉字
		# print(ACC_Time_List)

#用于决策树子空间
def Run_Model(train_attri,train_true_label,train_SubSpace_label,test_attri,test_label,DT_model=None,Leaf_To_SubSpaceIndex=None):
	SubSpace_Number=np.max(train_SubSpace_label)+1
	headline=""
	models=GetModules()

	All_Data_Predict_List=[]
	SubSpace_Predict_List=[]
	ACC_Time_List=[]
	ALL_Time=0
	#树模型
	for mi in range(5, 9):
		model_item = models[mi].items()
		model_name = ""
		Use_model = ""
		for key, values in model_item:
			model_name = key
			Use_model = values
		headline += (str(model_name) + "_" + str(SubSpace_Number)) + '\t'
		# print(x,SubSpace_Dic[x])
		# 总体的模型预测
		Use_model_All_Data = copy.deepcopy(Use_model)
		ALL_Data_Strat_Time = time.time()
		Use_model_All_Data.fit(train_attri, train_true_label)
		pre_l = Use_model_All_Data.predict(test_attri, )
		All_Data_End_Time = time.time()
		ALL_Data_Spent_Time = All_Data_End_Time - ALL_Data_Strat_Time
		All_Data_Predict_List.append(pre_l)
		All_Data_ACC = accuracy_score(test_label, pre_l)
		# 子空间预测
		# 统计时间
		Start_time = time.time()
		ssm=SSM(Use_model)

		ssm.fit(train_attri,train_true_label,train_SubSpace_label)
		# print(ssm.get_Pro_Each_SubSpace())
		test_pred=0
		if DT_model!=None:
			test_pred=ssm.predict_DecisionTree_SubSpace(test_attri,DT_model,Leaf_To_SubSpaceIndex)
		else:
			# pred_label=ssm.predict_Bp_DivideSubSpace(test_attri)
			test_pred = ssm.predict_withType(test_attri, train_SubSpace_label)
		SubSpace_ACC=accuracy_score(test_label,test_pred)
		Spent_time = time.time() - Start_time
		SubSpace_Predict_List.append(test_pred)  # 存储预测结果
		ALL_Time += Spent_time
		print(
			"Model:{} SubSapce_Nnumber:{} All_Data_ACC:{} SubSpaceACC:{} ALL_Data_Spent_Time:{} SubSpace_Spent_Time:{}".format(
				model_name, SubSpace_Number, All_Data_ACC, SubSpace_ACC, ALL_Data_Spent_Time, Spent_time))
		ACC_Time_List.append(np.array([ALL_Data_Spent_Time, All_Data_ACC, Spent_time, SubSpace_ACC]))
	# print(filename)
	train_attri = StandardScaler().fit_transform(train_attri)
	# test_attri = StandardScaler().fit_transform(test_attri)
	# 非树模型
	for mi in range(2, 5):
		model_item = models[mi].items()
		model_name = ""
		Use_model = ""
		for key, values in model_item:
			model_name = key
			Use_model = values
		headline += (str(model_name) + "_" + str(SubSpace_Number)) + '\t'
		# print(x,SubSpace_Dic[x])
		# 总体的模型预测
		Use_model_All_Data = copy.deepcopy(Use_model)
		ALL_Data_Strat_Time = time.time()
		Use_model_All_Data.fit(train_attri, train_true_label)
		pre_l = Use_model_All_Data.predict(StandardScaler().fit_transform(test_attri))
		All_Data_End_Time = time.time()
		ALL_Data_Spent_Time = All_Data_End_Time - ALL_Data_Strat_Time
		All_Data_Predict_List.append(pre_l)
		All_Data_ACC = accuracy_score(test_label, pre_l)
		# 子空间预测
		# filename = out_path + "SubSpace_" + SubSpace_Dic[x] + "_" + model_name
		# print(filename)
		# 统计时间
		Start_time = time.time()
		ssm = SSM(Use_model)
		ssm.fit(train_attri, train_true_label, train_SubSpace_label)
		test_pred = 0
		if DT_model != None:
			test_pred = ssm.predict_DecisionTree_SubSpace(test_attri, DT_model, Leaf_To_SubSpaceIndex)
		else:
			# pred_label=ssm.predict_Bp_DivideSubSpace(test_attri)
			test_pred = ssm.predict_withType(test_attri, train_SubSpace_label)
		SubSpace_ACC = accuracy_score(test_label, test_pred)
		Spent_time = time.time() - Start_time
		SubSpace_Predict_List.append(test_pred)  # 存储预测结果
		ALL_Time += Spent_time
		print(
			"Model:{} SubSapce_Nnumber:{} All_Data_ACC:{} SubSpaceACC:{} ALL_Data_Spent_Time:{} SubSpace_Spent_Time:{}".format(
				model_name, SubSpace_Number, All_Data_ACC, SubSpace_ACC, ALL_Data_Spent_Time, Spent_time))
		ACC_Time_List.append(np.array([ALL_Data_Spent_Time, All_Data_ACC, Spent_time, SubSpace_ACC]))
	return 	ACC_Time_List,np.column_stack(All_Data_Predict_List),np.column_stack(SubSpace_Predict_List),headline

def main(filename):
	# """测试数据Blob，Grid读数据"""
	i_path="D:\\sudty\\20190622集成研究\\"
	out_path="D:\\sudty\\20190622集成研究\双重模型结果\\"
	# filename="Grid_2V_Blods"
	data = pd.read_excel(i_path + "Data_集成_测试111.xlsx", sheet_name=filename)
	print(data.head())
	All_Data = np.array(data)
	train_attri = All_Data[:, 0:2]
	# print(train_attri[0:5])
	train_true_label = All_Data[:, 2]
	# print(train_true_label[0:5])
	test_attri=train_attri
	test_label=train_true_label
	SubSpace_Dic = {3: "400", 4: "225", 5: "100", 6: "49"}
	#"""MNIST读数据"""
	# All_Data = np.loadtxt(file_Path+filename+".txt", skiprows=1, delimiter='\t')
	# print("Load Data Over...")
	# train_attri = All_Data[:, 0:784]
	# # print(train_attri[0:5])
	# train_true_label = All_Data[:, 784]
	# # print(train_true_label)
	# SubSpace_Dic = {785: "49", 786: "100", 787: "225", 788: "400",789:"625"}
	# Test_Data=np.loadtxt(file_Path+"MNIST_test.txt",delimiter='\t')
	# test_attri = Test_Data[:,0:-1]
	# test_label = Test_Data[:,-1]
	#"""岩性数据读取"""
	# All_Data = np.loadtxt(file_Path+"", skiprows=1, delimiter='\t')
	# train_attri = All_Data[:, 0:3]
	# # print(train_attri[0:5])
	# train_true_label = All_Data[:, 3]
	# # print(train_true_label[0:5])
	# test_attri=train_attri
	# test_label=train_true_label
	# # # print(train_true_label[0:5])
	# # train_SubSpace_label=All_Data[:,8]
	# # # print(train_SubSpace_label[0:5])
	# # #测试数据
	# # test_data=np.loadtxt(test_data_path,skiprows=1,delimiter='\t')
	# # test_attri=test_data[:,2:8]
	# # test_label=test_data[:,8]
	# # print(test_attri[0:5])
	# # print(test_label[0:5])
	# # SubSpace_Dic={7:"400",8:"225",9:"100",10:"49"}
	# # SubSpace_Dic = {4: "400", 5: "225", 6: "100", 7: "49"}
	# # SubSpace_Dic = {7: "50", 8: "100", 9: "150", 10: "200",11:'250',12:"300",13:"350",14:"400"}
	# SubSpace_Dic = {4: "50", 5: "100", 6: "150", 7: "200", 8: '250', 9: "300", 10: "350", 11: "400"}

	# SubSpace_Dic = {7: "1450",8:"1600" ,9: "1225", 10: "900", 11: "625",12:"400",13:"225",14:"100",15:"49"}
	# SubSpace_Dic = {12: "400", 13: "225", 14: "100", 15: "49"}
	models=GetModules()

	All_Data_Predict_List=[]
	SubSpace_Predict_List=[]
	ACC_Time_List=[]
	headline=""
	ALL_Time=0
	#树模型
	for mi in range(5, 9):
		model_item = models[mi].items()
		model_name = ""
		Use_model = ""
		for key, values in model_item:
			model_name = key
			Use_model = values
		for x in SubSpace_Dic:
			headline+=(str(model_name)+"_"+str(SubSpace_Dic[x]))+'\t'
			# print(x,SubSpace_Dic[x])
			# 总体的模型预测
			Use_model_All_Data = copy.deepcopy(Use_model)
			ALL_Data_Strat_Time=time.time()
			Use_model_All_Data.fit(train_attri, train_true_label)
			pre_l = Use_model_All_Data.predict(test_attri,)
			All_Data_End_Time=time.time()
			ALL_Data_Spent_Time=All_Data_End_Time-ALL_Data_Strat_Time
			All_Data_Predict_List.append(pre_l)
			All_Data_ACC=accuracy_score(test_label,pre_l)
			# 子空间预测
			# filename = out_path + "SubSpace_" + SubSpace_Dic[x] + "_" + model_name
			# print(filename)
			train_SubSpace_label = All_Data[:, x]-1 #亚楠的数据是从1开始，所以要减1
			SubSpace_Number=int(SubSpace_Dic[x])
			#统计时间
			Start_time=time.time()
			ACC,pred_label=Run(train_attri, train_true_label, train_SubSpace_label,test_attri,test_label,Use_model,SubSpace_Number)

			Spent_time = time.time() - Start_time
			SubSpace_Predict_List.append(pred_label) #存储预测结果
			ALL_Time+=Spent_time
			print("Model:{} SubSapce_Nnumber:{} All_Data_ACC:{} SubSpaceACC:{} ALL_Data_Spent_Time:{} SubSpace_Spent_Time:{}".format(
				model_name,SubSpace_Dic[x],All_Data_ACC,ACC,ALL_Data_Spent_Time,Spent_time))
			ACC_Time_List.append(np.array([ALL_Data_Spent_Time,All_Data_ACC,Spent_time,ACC]))
	# print(filename)
	train_attri=StandardScaler().fit_transform(train_attri)
	test_attri=StandardScaler().fit_transform(test_attri)
	#非树模型
	for mi in range(3,5):
		model_item=models[mi].items()
		model_name=""
		Use_model=""
		for key,values in model_item:
			model_name=key
			Use_model=values
		for x in SubSpace_Dic:
			headline += (str(model_name) + "_" + str(SubSpace_Dic[x]))+'\t'
			# print(x,SubSpace_Dic[x])
			#总体的模型预测
			Use_model_All_Data=copy.deepcopy(Use_model)
			ALL_Data_Strat_Time = time.time()
			Use_model_All_Data.fit(train_attri, train_true_label)
			pre_l = Use_model_All_Data.predict(test_attri, )
			All_Data_End_Time = time.time()
			ALL_Data_Spent_Time = All_Data_End_Time - ALL_Data_Strat_Time
			All_Data_Predict_List.append(pre_l)
			All_Data_ACC = accuracy_score(test_label, pre_l)
			#子空间预测
			# filename=out_path+"SubSpace_"+SubSpace_Dic[x]+"_"+model_name
			train_SubSpace_label = All_Data[:, x]-1 #亚楠的数据是从1开始，所以要减1
			SubSpace_Number = int(SubSpace_Dic[x])
			Start_time = time.time()
			ACC,pred_label=Run(train_attri, train_true_label, train_SubSpace_label,test_attri,test_label,Use_model,SubSpace_Number)
			Spent_time = time.time() - Start_time
			SubSpace_Predict_List.append(pred_label)
			ALL_Time+=Spent_time
			print(
				"Model:{} SubSapce_Nnumber:{} All_Data_ACC:{} SubSpaceACC:{} ALL_Data_Spent_Time:{} SubSpace_Spent_Time:{}".format(
					model_name, SubSpace_Dic[x], All_Data_ACC, ACC, ALL_Data_Spent_Time, Spent_time))
			ACC_Time_List.append(np.array([ALL_Data_Spent_Time, All_Data_ACC, Spent_time, ACC]))
			# print(filename)
	All_Data_ALL_Result=np.column_stack(All_Data_Predict_List)
	np.savetxt(out_path+filename+"_All_Data_All_Result.txt",All_Data_ALL_Result,delimiter='\t',comments="",header=headline,fmt='%d')
	SubSpace_All_Result=np.column_stack(SubSpace_Predict_List)
	np.savetxt(out_path+filename+"_SubSpace_All_Result.txt",SubSpace_All_Result,delimiter='\t',comments="",header=headline,fmt='%d')
	ACC_Time_file=np.row_stack(ACC_Time_List)
	headline=headline.split('\t')[0:-1]
	ACC_Time_file=pd.DataFrame(ACC_Time_file,index=headline,columns=['All_Time','ALL_ACC','SubS_Time','SubS_ACC'])
	ACC_Time_file.to_csv(out_path + filename+"_ACC_Time_Pd.txt",'\t',index=True,header=True)#文件不能有汉字
	# np.savetxt(out_path + "ACC_Time_Pd.txt", ACC_Time_file, delimiter='\t', comments="", header=headline,
	# 		   fmt='%.06f')
	print("总共花费时间:",ALL_Time)

# #对Grid这个2维测试数据进行双重子空间预测
# def Run_Grid(filepath,filename):


if __name__=='__main__':
	# Get_ACC_Each_SubSpace(file_Path)
	list_name=["Grid_2V_Blods","Grid_2V_Circle","Grid_2V_Moon"]
	for x in list_name:
		main(x)
	# Get_Entropy()
	# main()
	# DT_SubSpace_Run()
