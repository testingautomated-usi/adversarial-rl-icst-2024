import pandas as pd
import argparse
import os



def calculate_coverage(file_name):
    #file_name = 'typed_2023-10-16_11-29-47_crashed_selected_typed_2023-10-16_15-50-01_crashed_selected_combined_grouping.csv'
    df = pd.read_csv(file_name)

    total_number_clusters = df['cluster'].nunique()

    agent_2 = df['agent_2_file'].unique()[0]
    random = df['random_file'].unique()[0]

    #assert agent_2 == 1
    #assert random == 1

    agent_2_failures = df[df['type'] == 'agent_2']['cluster'].unique()
    random_failures = df[df['type'] == 'random']['cluster'].unique()

    # Calculate coverage for 'agent_2'
    agent_2_coverage = df[df['type'] == 'agent_2']['cluster'].nunique()/total_number_clusters

    # Calculate coverage for 'random'   
    random_coverage = df[df['type'] == 'random']['cluster'].nunique()/total_number_clusters



    print(f"Coverage for 'agent_2': {agent_2_coverage} clusters")
    print(f"Coverage for 'random': {random_coverage} clusters")
    print(total_number_clusters)


    # Filter rows for 'agent_2' and 'random'
    agent_2_clusters = set(df[df['type'] == 'agent_2']['cluster'])
    random_clusters = set(df[df['type'] == 'random']['cluster'])

    all_clusters = set(df['cluster'])

    # Check if the clusters are complementary
    union_clusters = agent_2_clusters.union(random_clusters)
    intersection_clusters = agent_2_clusters.intersection(random_clusters)

    print(all_clusters)
    print(union_clusters)
    print(intersection_clusters)

    are_complementary = all_clusters - union_clusters
 
    agent_2_unique_failures = agent_2_clusters - intersection_clusters
    random_unique_failures = random_clusters - intersection_clusters

    agent_2_uniqueness = len(agent_2_unique_failures)/len(all_clusters)
    random_uniqueness = len(random_unique_failures)/len(all_clusters)

    if not union_clusters:
        print("The clusters of 'agent_2' and 'random' are complementary.")
        is_complementary= True
        overlap = False
    elif intersection_clusters == all_clusters:
        is_complementary = False
        overlap = True
        print("The clusters of 'agent_2' and 'random' completely overlaf")
        
    else:
        is_complementary = False
        overlap = False
        print("The clusters of 'agent_2' and 'random' are not complementary.")
        
    print(are_complementary)


    # Create a DataFrame with the information
    analysis_data = pd.DataFrame({
        'file_name': [file_name],
        'agent_2_coverage': [agent_2_coverage],
        'agent_2_unique_failures' : [len(agent_2_unique_failures)],
        'agent_2_uniqueness': [agent_2_uniqueness],
        'random_coverage': [random_coverage],
        'random_unique_failures' : [len(random_unique_failures)],
        'random_uniqueness' : [random_uniqueness],
        'number_of_clusters' : [len(all_clusters)],    
        'is_complementary': [is_complementary],
        'is_overlapping' : [overlap],
        'agent_2_failures' : [agent_2_failures],
        'random_failures' : [random_failures],
        'agent_2' : [agent_2],
        'random' : [random],
        
    })

    return analysis_data


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process CSV file and update analysis data")

    # Add an argument to capture the filename from the command line
    parser.add_argument("foldername", type=str, help="Name of the CSV file")

    # Parse the command line arguments
    args = parser.parse_args()

    folder_name = args.foldername

    files = os.listdir(folder_name)
    print(files)

    save_folder = '/code/rq_2/'

    save_file = os.path.join(save_folder, 'analysis.csv')
    for file in files:

        file_path = os.path.join(folder_name,file)
        analysis_data = calculate_coverage(file_path)


        # Check if 'analysis.csv' file exists
        if not os.path.isfile(save_file):
            # Create a new file with specified columns
            analysis_data.to_csv(save_file, index=False)
        else:
            # Append the row to the existing 'analysis.csv' file
            analysis_data.to_csv(save_file, mode='a', header=False, index=False)

