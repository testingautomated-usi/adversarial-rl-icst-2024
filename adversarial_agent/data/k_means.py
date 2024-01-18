import os
import argparse
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


def calculate_silhouette(feature_file, n_max = 50 ,cluster=None, percentage=20):

    df_1 = pd.read_csv(feature_file, index_col=0)
    df_2 = df_1.drop(columns= ['success','episode','type'])
    print(df_2)
    data = df_2.values

    if cluster:
        #kmeans_clusters = KMeans(n_clusters = cluster).fit_predict(data)
        kmeans_clusters = KMeans(n_clusters = cluster).fit(data)
        df_1['cluster'] = kmeans_clusters.labels_
        csv_file_name = os.path.splitext(feature_file)[0]
        file_name = csv_file_name + '_grouping' + '.csv'
        df_1.to_csv(file_name)
        print('file: ', file_name)
        return kmeans_clusters, data
     
    max_k_k = -1
    max_s_k = -1
    
    
    if len(data) < n_max:
         n_max = len(data)

    print('n_max: ', n_max)
    print("k-means")
    
    for k in range (2, n_max):
        kmeans_clusters = KMeans(n_clusters = k).fit_predict(data)
        s_k_means = silhouette_score(data,kmeans_clusters)
        print(k, s_k_means)

        increase = ((s_k_means - max_s_k)/abs(max_s_k))*100
        print('increase: ', increase)
        if increase >=percentage:
        #if s_k_means > max_s_k:
            max_s_k = s_k_means
            max_k_k = k
        print('max_s_k: ', max_s_k)

        return max_k_k, max_s_k
    '''    
    max_k_agglo = -1
    max_s_agglo = -1
    
    print("agglo")
    for k in range (2, n_max):
        agglomerative_clusters = AgglomerativeClustering(n_clusters=k, affinity="euclidean", linkage="complete").fit_predict(data)
        #agglomerative_clusters = AgglomerativeClustering(n_clusters=k).fit_predict(array_1)
        s_k_agglo = silhouette_score(data,agglomerative_clusters)
        print(k, s_k_agglo)

        increase = ((s_k_agglo - max_s_agglo)/abs(max_s_agglo))*100
        if increase >=percentage:
        #if s_k_agglo > max_s_agglo:
                max_s_agglo = s_k_agglo
                max_k_agglo = k
    
    print("max K-means: ")
    print("k: ", max_k_k)
    print("score: ", max_s_k)
    
    print("max aglo")
    print("k: ", max_k_agglo)
    print("score: ", max_s_agglo)
    '''
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering fails from crashed data")
    parser.add_argument("file_path", type=str, help="Path to the CSV file.")
    #parser.add_argument('-c', '--cluster', type=int, required=False)
    args = parser.parse_args()

    csv_file_name = args.file_path
    #c = args.cluster

    print(csv_file_name)
    #print(c)

    #csv_file_name = "./out/2023-07-27_19-39-38_crashed_selected.csv"
    #print(csv_file_name)
    k_clusters = calculate_silhouette(csv_file_name, cluster=c)
        
    
    

