from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import time
import sys




def test_hyperparams(data,labels):
    params ={"C":[0.0001,0.1,0.14,1.0,2.0,3.0],
    "gamma":[1e-1,1e-3,1e-4],
    "kernel":["rbf","linear","poly","sigmoid"],
    "coef0":[1.0,2.0,3.0],
    "degree":[1,4,5],
    "tol":[1e-4,1e-5]
    }
 
    init=time.time()
    recognizer = GridSearchCV(SVC(probability=True),params,n_jobs=-1)
    recognizer.fit(data["embeddings"],labels)
    print("[INFO] best hyperparameters: {}\n\n".format(recognizer.best_params_))
    end=time.time()
    print("[INFO] - Search complete, time spent: {:.2f} seconds".format(end - init))
    
    return recognizer

print("[INFO] - Loading embeddings")
data = pickle.loads(open("pickle/embeddings.pickle","rb").read())

print("[INFO] - Encoding labels")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
print("[INFO] - Training model")



if len(sys.argv) > 1:
    recognizer = test_hyperparams(data,labels)
else:
    recognizer = SVC(
            C=1.0,
            kernel="rbf",
            probability=True)


init = time.time()
recognizer.fit(data["embeddings"],labels)
end = time.time()
print("[INFO] - Training complete, {} seconds".format(end-init))
print("[INFO] - Saving data")
f = open("pickle/recog.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open("pickle/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
print("[INFO] - Data saved")

