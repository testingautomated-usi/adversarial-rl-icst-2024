import pandas as pd
import os
import argparse
import re

# Define the file paths
'''
random_name = 'typed_2023-10-16_22-16-57_crashed_selected.csv'

test_name = 'typed_2023-10-16_15-07-53_crashed_selected.csv'

file2_path = '/code/data/random_select/typed/' + random_name #random

file1_path = '/code/data/test_agent_2_select/typed/' + test_name #test agent 2
'''


def unify(file1_path,file2_path, t1, t2):
    # Load the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Concatenate the DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df['random_file'] = str(file1_path)
    combined_df['agent_2_file'] = str(file2_path)

    return combined_df

def get_subfolder_name(root_dir):
    # Initialize an empty list to store subdirectories
    subdirectories = []

    # Use os.walk to traverse through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # dirpath contains the path to the current directory
        # dirnames contains the names of the subdirectories in the current directory

        # Add the subdirectories to the list
        subdirectories.extend([os.path.join(dirpath, dirname) for dirname in dirnames])

    # Print the list of subdirectories
    return(subdirectories)


def get_subfolder_list(root_dir):
    items = os.listdir(root_dir)

    # Filter out only directories
    subdirectories = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]

    # Print the list of subdirectories
    #print(subdirectories)
    return subdirectories

if __name__ == '__main__':

    #parser = argparse.ArgumentParser(description="Filter and clean data from a CSV file.")
    #parser.add_argument("test_agent_2_folder", type=str, help="Path to the folder test agent 2 file.")
    #parser.add_argument("random_folder", type=str, help="Path to the folder random file.")
    #parser.add_argument('save_folder', type=str, help='path to the save file')

   
    save_folder = 'code/rq_2/combined_not_selected'
    base_test = 'code/rq_2/test_agent_2/'

    test_subfolders = get_subfolder_list(base_test)
    assert len(test_subfolders) == 10
    

    #i + '/filtered/selected/typed'
    base_random = '/code/rq_2/random/'
    random_subfolders = get_subfolder_list(base_random)
    assert len(random_subfolders) == 10

    for k in range((10)):
        #test_agent_2_folder = base_test + test_subfolders[k] + '/filtered/selected/typed'
        #random_folder = base_random + random_subfolders[k] + '/filtered/selected/typed'
        test_agent_2_folder = base_test + test_subfolders[k] + '/filtered/typed_not_selected'
        random_folder = base_random + random_subfolders[k] + '/filtered/typed_not_selected'
    
        #args = parser.parse_args()
        #test_agent_2_folder = args.test_agent_2_folder
        #random_folder = args.random_folder
        #save_folder = args.save_folder
        test_files = os.listdir(test_agent_2_folder)
        random_files = os.listdir(random_folder)

        assert len(test_files) == 10
        assert len(random_files) == 10

        pattern = r'run_(\d{8}-\d{6}_\d+)'
        
        for i in range(10):
            print('test file: ', test_files[i])
            test_file_path = os.path.join(test_agent_2_folder,test_files[i])
            print(test_file_path)   
            print('random file: ', random_files[i])
            random_file_path = os.path.join(random_folder,random_files[i])
            print(random_file_path)

            match_test = re.search(pattern, test_files[i])
            test_new_name = match_test.group(0)
            print('t: ',test_new_name)
            match_random = re.search(pattern, random_files[i])
            random_new_name = match_random.group(0)
            print('r: ', random_new_name)
            combined_file_name = f'{test_new_name}_{random_new_name}_combined.csv'
            print('c: ', combined_file_name)
            
            df = unify(test_file_path,random_file_path, test_subfolders[k], random_subfolders[k])
            #print(df)
            
            save_file = os.path.join(save_folder,combined_file_name)
            
            df.to_csv(save_file)
            print(f'Combined CSV file saved as {save_file}')
            

    #print('t: ',test_files)
    #print('r: ',random_files)
'''
file1_name = os.path.splitext(os.path.basename(file1_path))[0]
file2_name = os.path.splitext(os.path.basename(file2_path))[0]



# Save the combined DataFrame to a new CSV file
combined_file_name = f'{file1_name}_{file2_name}_combined.csv'
combined_df.to_csv(combined_file_name, index=False)

print(f'Combined CSV file saved as {combined_file_name}')

'''