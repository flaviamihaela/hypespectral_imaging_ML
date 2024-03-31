from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sn
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import scikitplot as skplt
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn import metrics
from numpy import load
from sklearn.model_selection import GridSearchCV
sn.axes_style('whitegrid');

#Define Functions
def read_HSI():
  X = loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
  y = loadmat('Indian_pines_gt.mat')['indian_pines_gt']
  print(f"X shape: {X.shape}\ny shape: {y.shape}")
  return X, y

def extract_pixels(X, y):
  q = X.reshape(-1, X.shape[2])
  df = pd.DataFrame(data = q)
  df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis=1)
  df.columns= [f'band{i}' for i in range(1, 1+X.shape[2])]+['class']
  df.to_csv('Dataset.csv')
  return df


#Read data
X, y = read_HSI()

fig = plt.figure(figsize = (12, 6))

for i in range(1, 1+6):
    fig.add_subplot(2,3, i)
    q = np.random.randint(X.shape[2])
    plt.imshow(X[:,:,q], cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f'Band - {q}')
plt.savefig('IP_Bands.png')

plt.figure(figsize=(10, 8))
plt.imshow(y, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.savefig('IP_GT.png')
#plt.show()

#Extract pixels and include them in a dataframe 
df = extract_pixels(X, y)
print(df.head())
print(df.info())
print(df.iloc[:, :-1].describe())

#Apply feature extraction methods:PCA, LDA, LLE, Isomap
number=200


#Apply PCA

pca= PCA(n_components=number)
fe=pca.fit_transform(df.iloc[:, :-1].values)

#Calculate explained variance

ev=pca.explained_variance_ratio_
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(ev))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


'''
#Apply k-PCA

kpca = KernelPCA(n_components = number, kernel='poly', eigen_solver='arpack')
fe = kpca.fit_transform(df.iloc[:, :-1].values)
'''

'''
#Apply LLE
lle= LocallyLinearEmbedding(n_neighbors=15, n_components=10)
fe=lle.fit_transform(df.iloc[:,:-1].values)
'''

'''
#Apply Isomap

'''


#Feature Extracted DataFrame

q = pd.concat([pd.DataFrame(data = fe), pd.DataFrame(data = y.ravel())], axis = 1)
q.columns = [f'FE-{i}' for i in range(1,201)]+['class']

print(q.head())


#plt.savefig('IP_PCA_Bands.png')

#q.to_csv('IP_40_PCA.csv', index=False)

x = q[q['class'] != 0]

X = x.iloc[:, :-1].values

y = x.loc[:, 'class'].values 

names = ['Alfalfa',	'Corn-notill', 'Corn-mintill',	'Corn',		'Grass-pasture','Grass-trees',
'Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill',
'Soybean-clean', 'Wheat',	'Woods',	'Buildings Grass Trees Drives',	'Stone Steel Towers']




#SVM classification

size=0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
svm =  SVC( C=1000, gamma='scale', kernel='rbf')
svm.fit(X_train, y_train)
ypred = svm.predict(X_test)



data = confusion_matrix(y_test, ypred)
df_cm = pd.DataFrame(data, columns=np.unique(names), index = np.unique(names))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,8))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Reds", annot=True,annot_kws={"size": 16}, fmt='d')
plt.savefig('cmap.png', dpi=300)

print(classification_report(y_test, ypred, target_names = names))

l=[]
for i in range(q.shape[0]):
  if q.iloc[i, -1] == 0:
    l.append(0)
  else:
    l.append(svm.predict(q.iloc[i, :-1].values.reshape(1, -1)))
clmap = np.array(l).reshape(145, 145).astype('float')
plt.figure(figsize=(10, 8))
plt.imshow(clmap, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.savefig('IP_cmap.png')
plt.show()


