from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

data=fetch_olivetti_faces()
features=data.data
targets=data.target

#visualize
# fig, subplots=plt.subplots(nrows=5, ncols=8, figsize=(14,8))
# subplots=subplots.flatten()

# #images of 40 different people
# for unique_user_id in np.unique(targets):
#     image_index=unique_user_id*8
#     subplots[unique_user_id].imshow(features[image_index].reshape(64,64),cmap='gray')
#     subplots[unique_user_id].set_xticks([])
#     subplots[unique_user_id].set_yticks([])
# #plt.show()

# #images of person no 1
# #first 10 images
# for j in range(10):
#     subplots[j].imshow(features[j].reshape(64,64),cmap='gray')
#     subplots[unique_user_id].set_xticks([])
#     subplots[unique_user_id].set_yticks([])
# #plt.show() 


#split data
X_train, X_test, y_train, y_test=train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)


#to find optimal no of eigenvectors (principle components)
pca=PCA(n_components=100, whiten=True)
# pca.fit(features)
# plt.figure(1,figsize=(12,8))
# plt.plot(pca.explained_variance_,linewidth=2)
# plt.xlabel('Components')
# plt.ylabel('Explained Variances')
# plt.show()

#########----------pca(principle component analysis). 

# PCA, is a dimensionality reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

# in this, we try to find eigenvectors - feature or pca component and eigenvalues - importance of a feature

#####in face recognition, eigenvectors are called eigenfaces
pca.fit(X_train)
X_pca=pca.transform(features)
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)


# #visualize pca
# number_of_eigenfaces=len(pca.components_)
# eigen_faces=pca.components_.reshape((number_of_eigenfaces, 64,64))

# fig,sub_plots=plt.subplots(nrows=10, ncols=10, figsize=(15,15))
# sub_plots=sub_plots.flatten()

# for i in range(number_of_eigenfaces):
#     sub_plots[i].imshow(eigen_faces[i], cmap="gray")
#     sub_plots[i].set_xticks([])
#     sub_plots[i].set_yticks([])
# plt.show()


models=[("Logistic Regression", LogisticRegression()), ("Support Vector Machine", SVC(C=1,probability=True)), ("Naive Bayes Classifer", GaussianNB())]

for name, model in models:
#using cross validation
    kfold=KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores=cross_val_score(model, X_pca, targets,cv=kfold)
    print("Mean of cross validation scores of ",model," :", cv_scores.mean())

#without using cross-validation
    # classifier_model=model
    # classifier_model.fit(X_train_pca, y_train)
    # y_predict=classifier_model.predict((X_test_pca))
    # print("Results with %s"%name)
    # print("Accuracy socre: %s"%(metrics.accuracy_score(y_test,y_predict)))