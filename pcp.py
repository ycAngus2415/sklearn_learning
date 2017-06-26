import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data[:, 0:2]
y = iris.target
n_features = X.shape[1]

for key, value in iris.items():
    try:
        print(key, value.shape)
    except:
        print(key)

C = 1.0

classifiers = {'L1 logistic': LogisticRegression(C=C, penalty='l1'),
               'L2 logistic(ovR)': LogisticRegression(C=C, penalty='l2'),
               'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                                 random_state=0),
               'L2 logistic(Multinomial)': LogisticRegression(
                   C=C, solver='lbfgs', multi_class='multinomial')
               }

n_classifiers = len(classifiers)

xx, yy =np.meshgrid(np.linspace(3, 9, 100), np.linspace(1, 5, 100).T)
Xfull = np.c_[xx.ravel(), yy.ravel()]


fig = plt.figure()

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
    print("classif_rate for %s : %f"%(name, classif_rate))

    probas = classifier.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    for k in range(n_classes):
        plt.subplot(n_classifiers, n_classes, index * n_classes + k +1)
        plt.title("class %d"%k)
        if k == 0:
            plt.ylabel(name)
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(3, 9, 1, 5), origin='lower')
        plt.xticks(())
        plt.yticks(())
        idx = (y_pred == k)
        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='k')
ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title('probability')
plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
plt.show()
