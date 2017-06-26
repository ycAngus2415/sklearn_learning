import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import sklearn as sk
from sklearn import svm
from sklearn import neighbors



iris = datasets.load_iris()
digits = datasets.load_digits()
#plt.imshow(digits.images[0], cmap='gray')
#plt.show()
data = digits.images.reshape((digits.images.shape[0],-1))
clf = svm.LinearSVC()
clf.fit(iris.data, iris.target)
predict = clf.predict([[5.0, 3.6, 1.3, 0.25]])
print(predict)
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)
predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print(predict)
perm = np.random.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
knn.fit(iris.data[:100], iris.target[:100])
pre = knn.score(iris.data[100:], iris.target[100:])
print(pre)
plt.scatter(iris.data[:,0],iris.data[:,1], c= iris.target)
plt.show()
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(iris.data[:,0], iris.data[:,1], iris.data[:,2], c=iris.target)

plt.show()
