import os
import argparse
import pandas as pd



def filter_by_success(csv_file, success_value):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file, index_col=0)

    # Filter the DataFrame based on the 'Success' column
    filtered_df = df[df['success'] == success_value]

    return filtered_df

def add_column_names(csv_file):
    col_names = list(range(120))+ ['success', 'episode']
    df = pd.read_csv(csv_file, names = col_names)
    df.to_csv(csv_file)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter and clean data from a CSV file.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file.")
    parser.add_argument("-a", "--add", action="store_true", help="Perform data cleaning.")
    #parser.add_argument('save_path', type=str, help="Path to save the new CSV file.")

    args = parser.parse_args()
    folder_name = args.file_path
    #save_name = args.save_path
    #files = os.listdir(folder_name)
    #csv_file_name = args.file_path
    add_data = args.add

    #if not os.path.exists(save_name):
    #    os.makedirs(save_name)
    
    success_value_to_filter = False

    for dirpath, dirnames, filenames in os.walk(folder_name):

        for filename in filenames:

            if filename.endswith('.csv'):
                
                full_path = os.path.join(dirpath, filename)
                
                print(full_path)
                
                csv_file_name = os.path.splitext(filename)[0]
                new_file_name = csv_file_name + '_crashed' + '.csv'
                new_save_folder = os.path.join(dirpath, 'filtered')
                print('nsf: ', new_save_folder)
                os.makedirs(new_save_folder, exist_ok=True)

                new_file_name = os.path.join(new_save_folder,new_file_name)

                print('new file name: ', new_file_name)
                print('dirpath: ', dirpath)
                
                if add_data:
                    
                    add_column_names(full_path)
                
                # Call the function to filter the DataFrame based on the 'Success' column
                filtered_data = filter_by_success(full_path, success_value_to_filter)
                
                #csv_file_name = os.path.splitext(filename)[0]
                #file_name = csv_file_name + '_crashed' + '.csv'
                #print('filename: ', file_name)

                #print(filtered_data)
                filtered_data.to_csv(new_file_name)
                