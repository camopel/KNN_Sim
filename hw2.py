from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from scipy.sparse import csc_matrix
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


train_dataset = pd.read_csv('./train.dat', sep='\t', encoding='utf-8', names=['rate','review']).dropna()
with open("./test.dat", "r") as f:
    test_dataset = f.readlines()
    f.close()
docs_train, docs_test, y_train, y_test = train_test_split(train_dataset.review, train_dataset.rate, test_size=0.25)
tfidf = TfidfVectorizer(min_df=0, max_df=0.2,ngram_range=(1,3))
X=tfidf.fit_transform(docs_train)
Xt=tfidf.transform(docs_test)

save_sparse_csr("docs_train.npz",X)
save_sparse_csr("docs_test.npz",Xt)

save_object_tofile("y_train.csv",y_train.values)
save_object_tofile("y_test.csv",y_test.values)

tmpe = Xt.dot(X.T)
save_sparse_csr("similarity.npz",tmpe)
del tmpe

E=0.005
K=500

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
            dist = csc_matrix.sum(csc_matrix.power(Xt[i]-X[j],2))
            w.append(1/dist)
            r.append(y_train.values[j])#y_train.index
            e.append(ve.data[m])    
    l.append(heapq.nlargest(K, zip(w,r,e), key=lambda s: s[0]))
    del r,w,e,ve
save_object_tofile("tuple_list.dat",l)
stop = timeit.default_timer()
print("Time:",stop-start)
#print(load_objec_fromfile("tuple_list.dat"))
#print(l)