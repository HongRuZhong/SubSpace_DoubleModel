import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.base as model_base
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import copy

import warnings
warnings.filterwarnings("ignore")

class SSM():
	def __init__(self,model):
		self.__module__ = "SOM_SubSpace_DoubleModels"
		self.model=model
	#计算每个子空间的熵
	def Get_Entropy_Each_SubSpace(self,train_data,train_target,train_subspace_target):
		self.label_num=np.max(train_target)+1#通过计算类别最大值作为类别的个数，注意从0开始，所以要加1
		self.SubSpace_data_array, self.SubSpace_target_array = self.split_subspace(train_data, train_target,
																				   train_subspace_target)
		self.conut_Each_SubSpace, self.Pro_Each_SubSpace = self.Gei_Label_Number_Probility(self.SubSpace_target_array,self.label_num)
		#计算每个子空间的熵值
		All_Entropy_List=[]
		AVE_Entropy_List=[]
		for x in self.Pro_Each_SubSpace:
			label_number=0.0
			Entropy_value=0.0
			for y in x:
				if y!=0:
					y=y+0.0001
					Entropy_value+=(-y*np.log(y))
					label_number+=1
			# print(label_number)
			All_Entropy=Entropy_value
			# print(All_Entropy)
			All_Entropy_List.append(All_Entropy)
			# print(All_Entropy_List[0])
			if label_number!=0:
				Ave_Entropy=Entropy_value/label_number
				# print(Ave_Entropy)
				AVE_Entropy_List.append(Ave_Entropy)
				# print(AVE_Entropy_List[0])
			else:
				Ave_Entropy=0
				# print(0)
				AVE_Entropy_List.append(Ave_Entropy)
		return np.row_stack(All_Entropy_List),np.row_stack(AVE_Entropy_List)
	#将子空间划分
	def split_subspace(self,data,label,sunspace):
		SubSpace_Number=np.max(sunspace)+1
		# 分开子空间
		self.SubSpace_data_list = []
		self.SubSpace_target_list = []
		for i in range(int(SubSpace_Number)):  # 类别从1开始就是1到401，从0开始是0到400
			self.SubSpace_data_list.append(
				[data[x] for x in range(len(sunspace)) if sunspace[x] == i])
			self.SubSpace_target_list.append(
				[label[x] for x in range(len(sunspace)) if sunspace[x] == i])
		# 子空间内部合并为numpy矩阵
		self.SubSpace_data_array = []
		self.SubSpace_target_array = []
		for x in range(len(self.SubSpace_data_list)):
			if len(self.SubSpace_data_list[x]) == 0:
				self.SubSpace_data_array.append(np.array([]))
				self.SubSpace_target_array.append(np.array([]))
			else:
				self.SubSpace_data_array.append(np.vstack(self.SubSpace_data_list[x]))
				self.SubSpace_target_array.append(np.vstack(self.SubSpace_target_list[x]))
		return self.SubSpace_data_array,self.SubSpace_target_array

	def Gei_Label_Number_Probility(self,label,label_number):
		label_number=int(label_number)
		# 计算每个子空间内部的真实类别的个数
		self.conut_Each_SubSpace = []
		for x in label:
			x = x.reshape(-1)
			if (len(x) == 0):
				self.conut_Each_SubSpace.append(np.zeros(label_number, np.int))
			else:
				num_t = np.bincount(x.astype(np.int), minlength=label_number)
				self.conut_Each_SubSpace.append(num_t)
		# print("self.conut_Each_SubSpace", self.conut_Each_SubSpace[0:5])
		# 计算每个子空间内部的真实类别的概率
		self.conut_Each_SubSpace = np.vstack(self.conut_Each_SubSpace)
		self.Pro_Each_SubSpace = np.zeros(self.conut_Each_SubSpace.shape, np.float64)
		for i in range(0, self.conut_Each_SubSpace.shape[0]):
			for j in range(self.conut_Each_SubSpace.shape[1]):
				self.Pro_Each_SubSpace[i, j] = self.conut_Each_SubSpace[i, j] * 1.0 / np.sum(
					self.conut_Each_SubSpace[i])
		return self.conut_Each_SubSpace,self.Pro_Each_SubSpace
	#fit
	def fit(self,train_data,train_target,train_subspace_target):
		self.label_num = np.max(train_target) + 1  # 通过计算类别最大值作为类别的个数，注意从0开始，所以要加1
		#用作划分子空间的类别的模型
		self.pred_model=DecisionTreeClassifier()
		self.pred_model.fit(train_data,train_subspace_target)
		#得到子空间数据
		self.SubSpace_data_array,self.SubSpace_target_array=self.split_subspace(train_data,train_target,train_subspace_target)
		self.conut_Each_SubSpace,self.Pro_Each_SubSpace=self.Gei_Label_Number_Probility(self.SubSpace_target_array,self.label_num)
		self.models = []
		#训练模型
		self.subspace_acc=[]
		for i in range(0, len(self.SubSpace_data_array)):
			t_d = self.SubSpace_data_array[i]
			label=self.SubSpace_target_array[i]
			#神经元没有数据，跳过处理，赋值为-1
			if len(label)==0:
				self.models.append(-1)
				self.subspace_acc.append(0)
				continue
			p_d = self.Pro_Each_SubSpace[i]
			if np.max(p_d) < 1:
				train_attri = t_d
				train_label = label
				self.The_Model=copy.deepcopy(self.model)
				self.The_Model.fit(train_attri, train_label)
				#计算子空间的拟合精度
				subspace_pred=self.The_Model.predict(train_attri)
				self.subspace_acc.append(accuracy_score(train_label,subspace_pred))
				# pred = clf.predict(train_attri)
				self.models.append(self.The_Model)
			else:
				self.models.append(label[0])
				self.subspace_acc.append(1)
	# 返回吗每个子空间内部的真实类别的个数
	def get_conut_Each_SubSpace(self):
		return self.conut_Each_SubSpace

	def get_SubSpace_ACC(self):
		return np.row_stack(self.subspace_acc)

	# 返回吗每个子空间内部的真实类别的概率
	def get_Pro_Each_SubSpace(self):
		return self.Pro_Each_SubSpace

	def calEuclideanDistance(self,vec1, vec2):
		dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
		return dist

	def predict_Bp_DivideSubSpace(self,test_data):
		give_type=self.pred_model.predict(test_data)
		# print("预测数据完成...")
		give_type = give_type.copy().astype(int)
		self.test_SubSpace_index = []
		self.pred_label_list = []
		for x in range(len(test_data)):
			SubSpace_label = give_type[x]
			if type(self.models[SubSpace_label]) == np.ndarray or type(self.models[SubSpace_label]) == int:
				self.pred_label_list.append(self.models[SubSpace_label])
			else:
				test_model = self.models[SubSpace_label]
				d = test_data[x].reshape(1, -1)  # data
				pred_label = test_model.predict(d)
				self.pred_label_list.append(pred_label)
		return np.vstack(self.pred_label_list)
	#给定type的情况下直接预测
	def predict_withType(self,test_data,give_type):
		#看看model
		# for x in range(len(self.models)):
		# 	print("第{}个模型为：{}".format(x,self.models[x]))
		give_type=give_type.copy().astype(int)
		self.pred_label_list = []
		for x in range(len(test_data)):
			SubSpace_label = give_type[x]
			# if SubSpace_label!=0:
			# 	continue
			# print("test_label:", SubSpace_label)
			# print("test_attri",test_data[x])
			# print(SubSpace_label)
			# print(type(self.models[test_label]))
			# print(len(self.models))
			if type(self.models[SubSpace_label]) == np.ndarray or type(self.models[SubSpace_label]) == int:
				self.pred_label_list.append(self.models[SubSpace_label])
			else:
				test_model = self.models[SubSpace_label]
				# print(x.shape)
				d = test_data[x].reshape(1, -1)#data
				# print("d", d)
				# print(x.shape)
				pred_label = test_model.predict(d)
				# print("预测结果:",pred_label)
				# print("pred_label",pred_label)
				self.pred_label_list.append(pred_label)
		return np.vstack(self.pred_label_list)
	#传入划分子空间的模型，来为测试数据分配子空间，最后一个参数是因为决策树划分的叶子节点不是0，1,2...n这种形式，所以传入映射字典
	def predict_DecisionTree_SubSpace(self,test_attri,DT_Model,Leaf_MapTo_SubSpaceIndex):
		self.pred_label_list=[]
		# print(Leaf_MapTo_SubSpaceIndex)
		standard_test=StandardScaler().fit_transform(test_attri)
		for x in range(len(test_attri)):
			Nostrad_data = test_attri[x].reshape(1, -1)  # data
			DT_Label=int(DT_Model.apply(Nostrad_data))
			# print(DT_Label)
			SubSpace_label = Leaf_MapTo_SubSpaceIndex[DT_Label]
			d=standard_test[x].reshape(1,-1)
			# print(SubSpace_label)
			if type(self.models[SubSpace_label]) == np.ndarray or type(self.models[SubSpace_label]) == int:
				self.pred_label_list.append(self.models[SubSpace_label])
			else:
				test_model = self.models[SubSpace_label]
				pred_label = test_model.predict(d)
				self.pred_label_list.append(pred_label)
		return np.vstack(self.pred_label_list)

	def predict(self,test_data,SOM_NNParams):
		self.test_SubSpace_index=[]
		self.pred_label_list=[]
		for x in test_data:
			test_subspace=[]
			for y in SOM_NNParams:
				# print("x,y", x, y)
				test_subspace.append(self.calEuclideanDistance(x,y))
			test_subspace=np.vstack(test_subspace)
			test_subspace=test_subspace.reshape(-1)
			# print("test_subspace",test_subspace)
			test_label=np.argmin(test_subspace)
			self.test_SubSpace_index.append(test_label)
			# print(test_label)
			# print(type(self.models[test_label]))
			if type(self.models[test_label])==np.ndarray or type(self.models[test_label])==int:
				self.pred_label_list.append(self.models[test_label])
			else:
				test_model = self.models[test_label]
				# print(x.shape)
				x=x.reshape(1,-1)
				# print(x.shape)
				pred_label = test_model.predict(x)
				# print("pred_label",pred_label)
				self.pred_label_list.append(pred_label)
		return np.vstack(self.pred_label_list)
	def get_Test_SubSpace_Index(self):
		return self.test_SubSpace_index