import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
from torchvision.datasets import CIFAR10

trainset = CIFAR10(root='./data', train=True, download=True)
testset = CIFAR10(root='./data', train=False, download=True)

X_train = np.array(trainset.data)  # (50000, 32, 32, 3)
y_train = np.array(trainset.targets)
X_test = np.array(testset.data)    # (10000, 32, 32, 3)
y_test = np.array(testset.targets)

print("Αρχικά σχήματα:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

print("\nΜετά flatten:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

print(f"\nΧρησιμοποιούνται 50000 train και 10000 test δείγματα.")

knn1 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=3)
ncc = NearestCentroid()

print("\nΕκπαίδευση KNN-1...")
knn1.fit(X_train, y_train)

print("Εκπαίδευση KNN-3...")
knn3.fit(X_train, y_train)

print("Εκπαίδευση Nearest Centroid...")
ncc.fit(X_train, y_train)

y_pred_knn1 = knn1.predict(X_test)
y_pred_knn3 = knn3.predict(X_test)
y_pred_ncc = ncc.predict(X_test)

acc_knn1 = accuracy_score(y_test, y_pred_knn1)
acc_knn3 = accuracy_score(y_test, y_pred_knn3)
acc_ncc = accuracy_score(y_test, y_pred_ncc)

print("\n===== ΑΠΟΤΕΛΕΣΜΑΤΑ =====")
print(f"KNN (1 neighbor):     {acc_knn1*100:.2f}%")
print(f"KNN (3 neighbors):    {acc_knn3*100:.2f}%")
print(f"Nearest Centroid:     {acc_ncc*100:.2f}%")
