import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import multiprocessing
from multiprocessing.pool import ThreadPool
from csv import DictWriter

#Entry in Dataset
def add(wcs,number_of_cluster):    
    # list of column names
    field_names = ['wcs2','wcs5','wcs7','wcs9','wcs12','number_of_cluster']
    dict={'wcs2':wcs[0],
          'wcs5':wcs[1],
          'wcs7':wcs[2],
          'wcs9':wcs[3],
          'wcs12':wcs[4],
          'number_of_cluster':number_of_cluster}
    
    with open('t.csv', 'a') as f_object:
     	dictwriter_object = DictWriter(f_object, fieldnames=field_names)
     	dictwriter_object.writerow(dict)
     	f_object.close()

#Finding Optimal number of cluster
def optimal_K(x):    
    l=list(x.iloc[0,:])
    
    # For finding categorical columns 
    categorical_columns=[]
    for i in range(0,len(l)):
        if type(l[i]) in [str,np.bool_]:
            categorical_columns.append(i)
    
    #Encoding of Categorical variable
    encoder = OrdinalEncoder()
    
    # transform data
    x[categorical_columns] = encoder.fit_transform(x.iloc[:,categorical_columns])
    if x[0][len(x)-1]==len(x):
        x.drop(x.columns[0],axis=1,inplace=True)
        
    # For Plotting the Elbow Method Curve
    wcss=[]
    wcs=[]
    for i in range(2, 13):  
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 0, max_iter=300)  
        kmeans.fit(x)  
        wcss.append(kmeans.inertia_)
        if i in [2,5,7,9,12]:
            wcs.append(kmeans.inertia_)
    plt.plot(range(1, 12), wcss,'o-')  
    plt.title('The Elobw Method Graph')  
    plt.xlabel('Number of clusters(k)')  
    plt.ylabel('wcss')  
    plt.show()  
    
    #Using Silhoutte Analysis
    score=[]
    for i in range(2,12):
        kmeans = KMeans(n_clusters=i, 
                        init='k-means++', 
                        random_state= 0, 
                        max_iter=300,
                        n_init=10)  
        kmeans.fit_predict(x)
        score.append(silhouette_score(x, kmeans.labels_, metric='euclidean'))
    print("Number of cluster = ",score.index(max(score))+2)
    add(wcs,score.index(max(score))+2)
    
    
# For Supervised datasets
from pathlib import Path
import os
from zipfile import ZipFile
current_directory=os.getcwd()
list_of_folders=os.listdir(current_directory)
print(list_of_folders)
dataset=[]
for folder in list_of_folders:
    try:
        f=current_directory+"/"+folder
        with open(f+"/"+folder+'.dat') as data1:
            f1=data1.read()
            d=str(f1)
            cnt=d.count('@')
            dataset2= pd.read_csv(f+"/"+folder+".dat",skiprows=cnt,header=None)
            dataset2.drop(dataset2.columns[-1],axis=1,inplace=True)
            dataset.append(dataset2)
    except:
        pass
dataset.sort(key=len)   

#Extracting Datasets of unsupervised data
from pathlib import Path
import os
from zipfile import ZipFile
current_directory=os.getcwd()
list_of_folders=os.listdir(current_directory)
dataset=[]
for folder in list_of_folders:
    for files in Path(folder).glob("*.*zip"):
        s=[]
        with ZipFile(files, 'r') as data:
            s=data.namelist()
            with data.open(s[0]) as data1:
                f1=data1.read()
                d=str(f1)
                cnt=d.count('@')
            try:
                dataset2= pd.read_csv(data.open(s[0]),skiprows=cnt,header=None)
                dataset.append(dataset2)
            except:
                pass


try:
    p1=multiprocessing.Process(target=(optimal_K), args=(dataset[0], ))
    p2=multiprocessing.Process(target=(optimal_K), args=(dataset[1], ))
    p3=multiprocessing.Process(target=(optimal_K), args=(dataset[2], ))
    p4=multiprocessing.Process(target=(optimal_K), args=(dataset[3], ))
    p5=multiprocessing.Process(target=(optimal_K), args=(dataset[4], ))
    p6=multiprocessing.Process(target=(optimal_K), args=(dataset[5], ))
    p7=multiprocessing.Process(target=(optimal_K), args=(dataset[6], ))
    p8=multiprocessing.Process(target=(optimal_K), args=(dataset[7], ))
    p9=multiprocessing.Process(target=(optimal_K), args=(dataset[8], ))
    p10=multiprocessing.Process(target=(optimal_K), args=(dataset[9], ))
    p11=multiprocessing.Process(target=(optimal_K), args=(dataset[10], ))
    p12=multiprocessing.Process(target=(optimal_K), args=(dataset[11], ))
    p13=multiprocessing.Process(target=(optimal_K), args=(dataset[12], ))
    p14=multiprocessing.Process(target=(optimal_K), args=(dataset[13], ))
    p15=multiprocessing.Process(target=(optimal_K), args=(dataset[14], ))
    p16=multiprocessing.Process(target=(optimal_K), args=(dataset[15], ))
    p17=multiprocessing.Process(target=(optimal_K), args=(dataset[16], ))
    p18=multiprocessing.Process(target=(optimal_K), args=(dataset[17], ))
    p19=multiprocessing.Process(target=(optimal_K), args=(dataset[18], ))
    p20=multiprocessing.Process(target=(optimal_K), args=(dataset[19], ))
    p21=multiprocessing.Process(target=(optimal_K), args=(dataset[20], ))
    p22=multiprocessing.Process(target=(optimal_K), args=(dataset[21], ))
    p23=multiprocessing.Process(target=(optimal_K), args=(dataset[22], ))
    p24=multiprocessing.Process(target=(optimal_K), args=(dataset[23], ))
    p25=multiprocessing.Process(target=(optimal_K), args=(dataset[24], ))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    p15.start()
    p16.start()
    p17.start()
    p18.start()
    p19.start()
    p20.start()
    p21.start()
    p22.start()
    p23.start()
    p24.start()
    p25.start()
    
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    p15.join()
    p16.join()
    p17.join()
    p18.join()
    p19.join()
    p20.join()
    p21.join()
    p22.join()
    p23.join()
    p24.join()
    p25.join()
except:
    pass