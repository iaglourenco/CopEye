from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle


print("[INFO] - Loading embeddings")
data = pickle.loads(open("pickle/embeddings.pickle","rb").read())

print("[INFO] - Encoding labels")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] - Training model")
params ={"C":[0.0001,0.001,0.01,0.091,0.095,0.1,0.11,0.12,0.13,0.14,1.0],
    "gamma":[1e-1,1e-2,1e-3,1e-4],
    "kernel":["rbf","linear","poly","sigmoid"],
    "coef0":[1.0,2.0,3.0,4.0,5.0],
    "degree":[1,2,3,4,5,10]
    }

'''
recognizer = GridSearchCV(
    SVC(kernel="rbf",gamma="auto",probability=True),params,n_jobs=-1,pre_dispatch="2*n_jobs")
recognizer.fit(data["embeddings"],labels)
print("[INFO] best hyperparameters: {}".format(recognizer.best_params_))
'''

recognizer = SVC(C=0.0001,coef0=3.0,degree=10,gamma=0.1,kernel="poly",probability=True)
recognizer.fit(data["embeddings"],labels)

f = open("pickle/recog.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open("pickle/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()