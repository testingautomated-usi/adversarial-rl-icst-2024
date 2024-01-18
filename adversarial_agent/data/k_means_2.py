import os
import argparse
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


def calculate_silhouette(feature_file, n_max = 50, percentage=20):

    df_1 = pd.read_csv(feature_file, index_col=0)
    df_2 = df_1.drop(columns= ['success','episode','type','random_file', 'agent_2_file', 'Unnamed: 0' ])
    #df_2 = df_1.drop(columns= ['success','episode','type', 'Unnamed: 0'])
    #print(df_2)
    data = df_2.values
     
    max_k_k = -1
    max_s_k = -1
    
    
    if len(data) < n_max:
         n_max = len(data)

    print('n_max: ', n_max)
    print("k-means")
    
    for k in range (2, n_max):
        #kmeans_clusters = KMeans(n_clusters = k).fit_predict(data)
        kmeans = KMeans(n_clusters=k).fit(data)
        kmeans_clusters = kmeans.labels_
        s_k_means = silhouette_score(data,kmeans_clusters)
        print(k, s_k_means)

        increase = ((s_k_means - max_s_k)/abs(max_s_k))*100
        print('increase: ', increase)
        print('percentage :', percentage)
        if increase >=percentage:
        #if s_k_means > max_s_k:
            max_s_k = s_k_means
            max_k_k = k
    print('max_s_k: ', max_s_k)
    print('max_k_k: ', max_k_k)

    return max_k_k, max_s_k
    
def get_clusters(feature_file, cluster):

    df_1 = pd.read_csv(feature_file, index_col=0)
    df_2 = df_1.drop(columns= ['success','episode','type', 'random_file', 'agent_2_file', 'Unnamed: 0'])
    #print(df_2)
    data = df_2.values

    kmeans_clusters = KMeans(n_clusters = cluster).fit(data)
    df_1['cluster'] = kmeans_clusters.labels_
    #csv_file_name = os.path.splitext(feature_file)[0]
    #file_name = csv_file_name + '_grouping' + '.csv'
    #df_1.to_csv(file_name)
    #print('file: ', file_name)
    return df_1
    #return kmeans_clusters, data
    
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Clustering fails from crashed data")
    #parser.add_argument("file_path", type=str, help="Path to the folder with the combined data")
    #parser.add_argument('-c', '--cluster', type=int, required=False)
    #args = parser.parse_args()

    #folder_name = args.file_path
    
    #folder_name = 'code/rq_2_v2/combined/'
    #save_folder = 'code/rq_2_v2/cluster_data/'

    folder_name = '/code/rq_2/combined/'
    save_folder = '/code/rq_2/cluster_data/'

    #folder_name = '/code/rq_2/combined_not_selected'
    #save_folder = '/code/rq_2/cluster_data_not_selected'

    files = os.listdir(folder_name)
    print(files)
    print(len(files))
    
    k_list = []
    k_score = []
    for file in files:
        print(file)
        file_path = os.path.join(folder_name,file)
        print(file_path)
        
        k, score = calculate_silhouette(file_path, 50, 30)
        k_list.append(k)
        
        df_b = get_clusters(file_path, k)
        #print(df_b)

        save_file_name = os.path.join(save_folder,file)
        print('s: ', save_file_name)
        df_b.to_csv(save_file_name)
        
    dic = {'file': files, 'k': k_list}

    df_a = pd.DataFrame(dic)

    print(df_a)
    save_file = os.path.join(save_folder, 'cluster_info_not_selected.csv')
    df_a.to_csv(save_file)

    
    
    
    

        

    
    

