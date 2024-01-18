import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler



def get_clusters(file):
    # Load the trajectory data from CSV
    df_1 = pd.read_csv(file, index_col=0)
    df_2 = df_1.drop(columns= ['success','episode','type','random_file', 'agent_2_file', 'Unnamed: 0' ])
    
    data = df_2.values

    # Standardize the data (important for distance-based clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Initialize DBSCAN with appropriate epsilon and min_samples values
    epsilon = 0.1  # Adjust based on your dataset
    min_samples = 5  # Adjust based on your dataset

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

    # Fit the DBSCAN clustering model
    clusters = dbscan.fit_predict(X_scaled)

    # Add the cluster labels back to the original DataFrame
    df_1['cluster'] = clusters
    
    n_cluster = df_1['cluster'].nunique()

    # If you want to save the clustered data to a new CSV file
    #data.to_csv('clustered_trajectory_data.csv', index=False)
    return df_1, n_cluster
    
if __name__ == "__main__":
    folder_name = 'code/rq_2/combined/'
    save_folder = 'code/rq_2/dbscan/'
    
    files = os.listdir(folder_name)
    clusters = []
    for file in files:
        file_path = os.path.join(folder_name,file)
        data, c = get_clusters(file_path)
        clusters.append(c)
    
    save_file_name = os.path.join(save_folder, "dbscan"+file)
    data.to_csv(save_file_name)

    dic = {'file' : files, 'c': clusters}
    
    df_a = pd.DataFrame(dic)
    
    save_file = os.path.join(save_folder, 'dbscan_info.csv')
    df_a.to_csv(save_file)
