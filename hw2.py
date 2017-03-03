from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from scipy.sparse import csr_matrix
import pandas as pd
import timeit
import heapq
import numpy as np
import pickle

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'])

def save_object_tofile(filename,obj):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)
        fp.close()
def load_objec_fromfile(filename):
    with open (filename, 'rb') as fp:
        obj = pickle.load(fp)
        fp.close()
        return obj
def save_result(filename,yy):
    with open(filename,'w') as f:
        for i in yy:
            if i==1:
                f.write("+1\n")
            else:
                f.write("-1\n")
        f.close()    
    print("complete")
    
def print_test_result(yt,ypre):
	print(metrics.classification_report(yt,ypre,digits=4))
	print("\n")
	print(metrics.confusion_matrix(yt,ypre))
def load_test_data():
	with open("./test.dat", "r") as f:
		ls = f.readlines()
		f.close()
	return ls

def predict_analyse(filename):
	obj=load_objec_fromfile(filename)
	rst = []
	for tl in obj:
		sumr=0    
		minE=1
		count=K
		zero=0
		for t in tl:
			if count<=0:
				break;
			count-=1
			sumr+=t[0]*t[1]
			if t[2]<minE:
				minE=t[2]
		if sumr==0:
			zero=1
		if sumr>0:
			y=1
		else:
			y=-1
		rst.append([count,minE,zero,y])
	del obj
	return rst
	
E=0.01
K=500
	
train_dataset = pd.read_csv('./train.dat', sep='\t', encoding='utf-8', names=['rate','review']).dropna()
docs_train, docs_test, y_train, y_test = train_test_split(train_dataset.review, train_dataset.rate, test_size=0.25)
tfidf = TfidfVectorizer(min_df=0, max_df=0.2,ngram_range=(1,3))
X=tfidf.fit_transform(docs_train)
Xt=tfidf.transform(docs_test)
#test_dataset = load_test_data()
#Xt=tfidf.transform(test_dataset) 

#knn = KNeighborsClassifier(n_neighbors=300,weights='distance',algorithm='brute',n_jobs=1)
#knn.fit(X, y_train)
#y_predicted = knn.predict(Xt)

save_sparse_csr("docs_train.npz",X)
save_sparse_csr("docs_test.npz",Xt)
save_object_tofile("y_train.csv",y_train.values)
save_object_tofile("y_test.csv",y_test.values)
tmpe = Xt.dot(X.T)
save_sparse_csr("similarity.npz",tmpe)
del tmpe


start = timeit.default_timer()
l = []
for i in range(Xt.shape[0]):
    print(i)
    ve = Xt[i].dot(X.T)
    r = []
    w=[]
    e=[]
    for m in range(ve.getnnz()): 
        if ve.data[m]>=E:
            j=ve.indices[m]            
            dist = csr_matrix.sum(csr_matrix.power(Xt[i]-X[j],2))
            w.append(1/dist)
            r.append(y_train.values[j])#y_train.index
            e.append(ve.data[m])    
    l.append(heapq.nlargest(K, zip(w,r,e), key=lambda s: s[0]))
    del r,w,e,ve
save_object_tofile("tuple_list.dat",l)
stop = timeit.default_timer()
print("Time:",stop-start)

rst = predict_analyse("tuple_list.dat")
y_predicted = [i[3] for i in rst]
print_test_result(y_test,y_predicted)
#save_result("result.dat",y_predicted)