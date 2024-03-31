#Serial Fusion Independent
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sn
from scipy.io import loadmat
sn.axes_style('whitegrid');
import skimage.filters
import cv2
from skimage import io
from sklearn import pipeline
from sklearn import metrics
#Functions
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
#START

#Read dataset
X, y = read_HSI()
plt.imshow (X[:,:,8])


#Compute Gabor Features
copy_X=X;

#Gabor Kernels that will be applied for each 
kernel1=skimage.filters.gabor_kernel(frequency=0.2, theta=np.pi)
kernel2=skimage.filters.gabor_kernel(frequency=0.2, theta=np.pi/4)
kernel3=skimage.filters.gabor_kernel(frequency=0.2, theta=3*np.pi/4)
kernel4=skimage.filters.gabor_kernel(frequency=0.2, theta=np.pi/2)
#plt.imshow(np.real(kernel1))
#plt.imshow(np.real(kernel2))
#plt.imshow(np.real(kernel3))
#plt.imshow(np.real(kernel4))

#Generate gabor filter bank and store the real/imaginary filtered images 
num=1
df_r=pd.DataFrame()
df_i=pd.DataFrame()
frequency=0.2
for i in np.arange(copy_X.shape[2]):
    test = copy_X[:,:,i]
    for th in np.arange(0, np.pi, np.pi/4):
        r,i= skimage.filters.gabor(test, frequency=0.2, theta=th)
        r=r.reshape(-1)
        i=i.reshape(-1)
        df_r[num]= r
        df_i[num]=i
        num+=1
#print(df_r.head())
#print(df_i.head())
#print(df_r.iloc[:,:].describe())
#print(df_i.iloc[:,:].describe())
scaler = MinMaxScaler()
df_r_scaled=pd.DataFrame(scaler.fit_transform(df_r),columns=df_r.columns)
df_i_scaled=pd.DataFrame(scaler.fit_transform(df_i),columns=df_i.columns)
#print(df_r_scaled.head())
#print(df_i_scaled.head())

#Extract magnitude gabor features for each pixel 
df_r_squared=df_r_scaled.pow(2)
df_i_squared=df_i_scaled.pow(2)
df_squares_sum=df_r_squared+df_i_squared
#print(df_r_squared.head())
#print(df_i_squared.head())
#print(df_squares_sum.head())
df_magn_gabor=df_squares_sum.pow(1/2)
#print(df_magn_gabor.head())
df_mg=pd.concat([df_magn_gabor, pd.DataFrame(data = y.ravel())], axis=1)
df_mg.rename({0: "class"}, axis='columns', inplace =True)
#print(df_mg.head())
#print(df_mg.info())


#Extract pixels and add them to dataframe together with class label
df = extract_pixels(X, y)
#print(df.head())
#print(df.info())
#print(df.iloc[:, :-1].describe())


#Feature extraction algorithms
#pca=PCA(n_components=150)
#ica=FastICA(n_components=50)
#lle= LocallyLinearEmbedding(n_components=110, n_neighbors=300)
kpca=KernelPCA(n_components=130, kernel='linear')

#Apply feature extraction to spectral data
#principalComponents_s = pca.fit_transform(df.iloc[:, :-1].values)
#independentComponents_s = ica.fit_transform(df.iloc[:, :-1].values)
#lleComponents_s = lle.fit_transform(df.iloc[:, :-1].values)
kpcaComponents_s = kpca.fit_transform(df.iloc[:, :-1].values)

#Apply feature extraction to textural data
#principalComponents_g = pca.fit_transform(df_mg.iloc[:, :-1].values)
#independentComponents_g = ica.fit_transform(df_mg.iloc[:, :-1].values)
#lleComponents_g = lle.fit_transform(df_mg.iloc[:, :-1].values)
kpcaComponents_g = kpca.fit_transform(df_mg.iloc[:, :-1].values)

#Change ' data=' and 'range'
q = pd.concat([pd.DataFrame(data = kpcaComponents_s), pd.DataFrame(data = y.ravel())], axis = 1)
q.columns = [f'S-{i}' for i in range(1,130+1)]+['class']
#print(q.head())

#Change ' data=' and 'range'
gab = pd.concat([pd.DataFrame(data = kpcaComponents_g), pd.DataFrame(data = y.ravel())], axis = 1)
gab.columns = [f'G-{i}' for i in range(1,130+1)]+['class']
#print(gab.head())

#Fuse the extracted textural features and the extracted spectral features together

q=q.drop(['class'], axis=1)
#print(q.head())
#print(q.info())
gab=gab.drop(['class'], axis=1)
#print(gab.head())
df_fused=pd.concat([gab,q], axis=1)
#print(df_fused.head())
df_fused=pd.concat([df_fused, pd.DataFrame(data = y.ravel())], axis=1)
#print(df_fused.head())
df_fused.rename(columns={0: 'class'}, inplace=True)
#print(df_fused.head())


#Define the dataset before feeding it to the classification stage
x = df_fused[df_fused['class'] != 0]
X = x.iloc[:, :-1].values
y = x.loc[:, 'class'].values 
#print(X)
#print(y)
#print(X.shape)
#print(y.shape)

names = ['Alfalfa',	'Corn-notill', 'Corn-mintill',	'Corn',		'Grass-pasture','Grass-trees',
'Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill',
'Soybean-clean', 'Wheat',	'Woods',	'Buildings Grass Trees Drives',	'Stone Steel Towers']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)


#Classification algorithm
svm=SVC()

pipe = pipeline.Pipeline(steps=[('svm', svm)])

param_grid = {'svm__kernel':['rbf'], 'svm__C':[0.0001, 0.001, 1, 10, 100, 1000], 'svm__gamma':['scale']}

search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=2, scoring='accuracy')

#Fitting
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#Prediction
ypred=search.predict(X_test)

#Plot Confusion Matrix
data = confusion_matrix(y_test, ypred)
df_cm = pd.DataFrame(data, columns=np.unique(names), index = np.unique(names))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,8))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Reds", annot=True,annot_kws={"size": 16}, fmt='d')
plt.savefig('cmap.png', dpi=300)

print(classification_report(y_test, ypred, target_names = names))

print(metrics.f1_score(y_test, ypred, average='weighted', labels=np.unique(ypred)))
print(cohen_kappa_score(y_test, ypred))

