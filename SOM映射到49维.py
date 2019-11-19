import numpy as np
import My_PythonLib as mp
from sklearn.model_selection import train_test_split

filepath="D:\sudty\亚楠_SOM岩性子空间\\"
# filepath="D:\Data\Data_机器学习常用数据集\\"
def Run(ipath):
	data=np.loadtxt(ipath,delimiter='\t',)
	attri=data[:,0:-2]
	label=data[:,-2]
	print(attri.shape,label)
	# exit()
	train_attri,test_attri,train_label,test_label=train_test_split(attri,label,test_size=0.3,random_state=32)
	mp.ML_Model_Run_NoTree(train_attri,train_label,test_attri,test_label)
	# mp.ML_Model_Run(attri,label,attri,label)

if __name__=="__main__":
	Run(filepath+"考虑类别信息onehot的7x7SOM_标准化.txt")
