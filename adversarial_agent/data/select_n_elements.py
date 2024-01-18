import os
import argparse
import pandas as pd
import numpy as np

def select_n_elements(csv_file, elem):
    df = pd.read_csv(csv_file,index_col=0)
    #print(df)
    episodes = df['episode'].values
    sucess = df['success'].values
    #print('episodes: ', df['episode'])
    df = df.drop('episode', axis='columns')
    df = df.values
    i = 0

    new_data = []
    for el in df:
        i+=1
        ind = np.where(el == elem)[0][0]
        last = ind
        first = ind - 16
        if first < 16:
            last = 16
            first=0
        new_array = el[first:last]
        #new_array= np.append(new_array,el[-1])
        #print('new array: ', new_array)
        new_data.append(new_array)
    col = range(16)
    df_2 = pd.DataFrame(new_data,columns=col)
    df_2['success'] = sucess
    df_2['episode'] = episodes

    #csv_file = os.path.splitext(csv_file_name)[0]
    #file_name = csv_file + '_selected' + '.csv'
    #df_2.to_csv(file_name)
    return df_2
    
    print('results in: ', file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="select 4 last data points from the trajectory of the vehicles.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file.")
    #parser.add_argument('save_path', type=str, help="Path to save the new CSV file.")
    args = parser.parse_args()
    
    folder_name = args.file_path
    #save_name = args.save_path
    elem = 0.0

    for dirpath, dirnames, filenames in os.walk(folder_name):

        for filename in filenames:

            if filename.endswith('.csv') and 'crashed' in filename:

                full_path = os.path.join(dirpath, filename)    

                df = select_n_elements(full_path, elem)

                csv_file = os.path.splitext(filename)[0]
                new_file_name = csv_file + '_selected' + '.csv'
                new_save_folder = os.path.join(dirpath,'selected')
                os.makedirs(new_save_folder,exist_ok=True)

                new_file = os.path.join(new_save_folder,new_file_name)

                df.to_csv(new_file)
                print('results in: ', new_file)
                print('dirpath: ', dirpath)

