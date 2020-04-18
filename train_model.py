from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import time
import sys




def test_hyperparams():
    params ={"C":[0.0001,0.1,0.14,1.0],
    "gamma":[1e-1,1e-3,1e-4],
    "kernel":["rbf","linear","poly","sigmoid"],
    "coef0":[1.0,2.0,3.0],
    "degree":[1,4,5],
    "tol":[1e-4,1e-5],
    "decision_function_shape":["ovo","ovr"],
    }
 
    init=time.time()
    recognizer = GridSearchCV(SVC(probability=True),params,n_jobs=-1)
    recognizer.fit(data["embeddings"],labels)
    print("[INFO] best hyperparameters: {}\n\n".format(recognizer.best_params_))
    end=time.time()
    print("[INFO] - Search complete, time spent: {:.2f} seconds".format(end - init))
    print("[INFO] Setting hyperparameters: {}\n\n".format(recognizer.best_params_))
    recognizer = SVC(
        C=recognizer.best_params_["C"],
        coef0=recognizer.best_params_["coef0"],
        degree=recognizer.best_params_["degree"],
        gamma=recognizer.best_params_["gamma"],
        kernel=recognizer.best_params_["kernel"],
        tol=recognizer.best_params_["tol"],
        decision_function_shape=recognizer.best_params_["decision_function_shape"],
        probability=True)
    return recognizer

print("[INFO] - Loading embeddings")
data = pickle.loads(open("pickle/embeddings.pickle","rb").read())

print("[INFO] - Encoding labels")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
print("[INFO] - Training model")



if len(sys.argv) > 1:
    recognizer = test_hyperparams()

recognizer = SVC(
        C=1.0,
        coef0=3.0,
        degree=5,
        gamma=0.1,
        kernel="poly",
        tol=0.0001,
        decision_function_shape="ovo",
        probability=True)
recognizer.fit(data["embeddings"],labels)
print("[INFO] - Training complete")

#best hyperparameters: {'C': 1.0, 'coef0': 3.0, 'decision_function_shape': 'ovo', 'degree': 5, 'gamma': 0.1, 'kernel': 'poly', 'tol': 0.0001}

print("[INFO] - Saving data")
f = open("pickle/recog.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open("pickle/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
print("[INFO] - Data saved")

