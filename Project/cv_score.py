from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.load("data.npy")
X, y = data[:, :-1], data[:, -1]
#print(X.shape, y.shape)
# create loocv procedure
cv = LeaveOneOut()
# create model
#model = RandomForestClassifier(random_state=1)

#ANOVA Feature Selection and Linear SVM Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


anova_filter = SelectKBest(f_classif, k=5200)
clf = LinearSVC()
model = make_pipeline(StandardScaler(),anova_filter, clf)
  #model.fit(X_train, y_train)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
np.save("mean_scores_5200k.npy",np.array(mean(scores)))




data = np.load("data.npy")
X, y = data[:, :-1], data[:, -1]
#print(X.shape, y.shape)
# create loocv procedure
cv = LeaveOneOut()
# create model
#model = RandomForestClassifier(random_state=1)

#ANOVA Feature Selection and Linear SVM Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


anova_filter = SelectKBest(f_classif, k=1735)
clf = LinearSVC()
model = make_pipeline(StandardScaler(),anova_filter, clf)
  #model.fit(X_train, y_train)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
np.save("mean_scores_1735k.npy",np.array(mean(scores)))

