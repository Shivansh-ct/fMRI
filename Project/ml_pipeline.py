import pickle
import numpy as np
"""
from nilearn import datasets
from nilearn import input_data

power = datasets.fetch_coords_power_2011()
print('Power atlas comes with {0}.'.format(power.keys()))



coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T

print('Stacked power coordinates in array of shape {0}.'.format(coords.shape))


from sklearn import preprocessing, covariance
myScaler = preprocessing.StandardScaler()



arr = ["tfMRI_WM_","tfMRI_MOTOR_","tfMRI_LANGUAGE_","tfMRI_EMOTION_"]
final = ["LR","RL"]
sub=[100307,115017,115724,130316,144832,158843,159340,175237,202820,206828,257845,579665,654754,680957,786569,871762,894774,100408,115219,130013,130417,144933,159138,175035,191437,206525,256540,361941,579867,680250,724446,788674,894067,932554,100610,115320,130114,144731,145127,159239,175136,191841,206727,257542,365343,580044,680452,784565,792766,894673]

#sub1 = [100307, 115017]

cl = 0
#Calculating the Functional Connectivity Matrix for each bold file and stacking them together in "data" variable for further Machine Learning
for i in sub:
  for j in arr:
    for k in final:
     try:   
      path = "/mnt/scratch1/csy207576/project1/data3/"+str(i)+"/"+j+k+"/"+j+k+".nii.gz"
      print(path)
      spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, smoothing_fwhm=6, radius=10.,allow_overlap=True, detrend=True, standardize=True, low_pass=0.1, high_pass=0.01, t_r=2)
      timeseries = np.array(spheres_masker.fit_transform(path), dtype=np.float64)
      timeseries = myScaler.fit_transform(timeseries)
      timeseries = covariance.empirical_covariance(timeseries)
      timeseries = covariance.shrunk_covariance(timeseries, shrinkage=0.8)

  #print('time series has {0} samples'.format(timeseries.shape[0]))
      try:
        from sklearn.covariance import GraphicalLassoCV
      except ImportError:
      # for Scitkit-Learn < v0.20.0
        from sklearn.covariance import GraphLassoCV as GraphicalLassoCV

      covariance_estimator = GraphicalLassoCV(cv=3, verbose=1)
      covariance_estimator.fit(timeseries)
      matrix = covariance_estimator.covariance_
      np.save(str(i)+"_"+j+k+".npy", matrix)
      d=matrix[np.triu_indices(264, k = 1)]
      d = d.reshape(1,34716)
      d = np.concatenate((d,[[cl]]),axis=1)
      if i==100307 and j=="tfMRI_WM_" and k=="LR":
        data = d
      else:
        data = np.concatenate((data,d))
     except:
        continue
  cl = cl+1
    

print('Covariance matrix has shape {0}.'.format(matrix.shape),"saving data matrix")
np.save("data.npy",data)


data = np.load("data.npy")
#data = np.load("data.npy")
#Making the last column LabelEncoded
#data[:,data.shape[1]-1] = data[:,data.shape[1]-1]-1

#data[:,data.shape[1]-1]

#Separating the features and labels
X = data[:,:-1]
y = data[:, data.shape[1]-1]

#X.shape

#Generating the Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=None)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



#ANOVA Feature Selection and Linear SVM Pipeline

print("Printing metrics using Linear Kernel")
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

#Selecting top 5% features
anova_filter = SelectKBest(f_classif, k=1735)
clf = LinearSVC()
anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)


# save the model to disk
filename = 'anova_svm.sav'
pickle.dump(anova_svm, open(filename, 'wb'))


#Printing the Metrics
from sklearn.metrics import classification_report, accuracy_score

y_pred = anova_svm.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy is: ", accuracy_score(y_test, y_pred))


#Classification using ANOVA and rbf kernel

print("Printing metrics using rbf kernel")


from sklearn.svm import SVC
clf = SVC(gamma='auto')
anova_svm_rbf = make_pipeline(anova_filter, clf)
anova_svm_rbf.fit(X_train, y_train)


# save the model to disk
filename = 'anova_svm_rbf.sav'
pickle.dump(anova_svm_rbf, open(filename, 'wb'))


#Printing the Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_pred = anova_svm_rbf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy is: ", accuracy_score(y_test, y_pred))






#Drawing a graph for accuracy vs features
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
x=[]
y=[]

#Selecting top k features

for i in range(100,34717,100):
  x.append(i)
  anova_filter = SelectKBest(f_classif, k=i)
  clf = LinearSVC()
  anova_svm_plt = make_pipeline(anova_filter, clf)
  anova_svm_plt.fit(X_train, y_train)
  y_pred = anova_svm_plt.predict(X_test) 
  y.append(accuracy_score(y_test, y_pred))
  print(i)

#Printing the Metrics
#from sklearn.metrics import classification_report, accuracy_score

#y_pred = anova_svm_plt.predict(X_test)
#print(classification_report(y_test, y_pred))
#print("Accuracy is: ", accuracy_score(y_test, y_pred))
  np.save("x.npy", np.array(x))
  np.save("y.npy", np.array(y))
"""
x = np.load("x.npy")
y = np.load("y.npy")

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.xlabel('Number of features(k)')
plt.ylabel('Accuracy')
plt.title('Accuracy as a function of best k features (of FC Matrix)')
plt.savefig("output_100.jpg")
plt.show()
