import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt 

def read_pickle_to_dataframe(file_path, script_name) -> pd.DataFrame:
        script_name = os.path.basename(script_name)
        script_name = os.path.splitext(script_name)[0]
        file_string = f'{file_path}/{script_name}.pkl'

        with open(file_string, 'rb') as file: # read binary
            data = pickle.load(file)
        df = pd.DataFrame(data)
        return df, file_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read pickle file and convert it to pandas dataframe')
    parser.add_argument('--pickle_file_path', type=str, default='results_storage', help='The directory of the pickle file')
    parser.add_argument('--script_name', type=str, help='The name of the script')
    parser.add_argument('--view_all_columns', action='store_true')

    args = parser.parse_args()

    df, _ = read_pickle_to_dataframe(args.pickle_file_path, args.script_name)

    # Assuming you have a list of column names
    selected_columns = ['model',
                        'backbone_version',
                        'all']

    # Filter the selected_columns list to only include column names that exist in the DataFrame
    intersection_columns = [col for col in selected_columns if col in df.columns]

    # Select these columns from the dataframe
    selected_data = df[intersection_columns]

    # Print the selected data
    print(selected_data)

    if args.view_all_columns:
         print(df.tail())

    # Get metadata for the textbox
    '''
    first_row = df.iloc[0]
    meta_columns = [col for col in df.columns if col not in selected_columns]     
    info_dict = {col: first_row[col] for col in meta_columns}
    info_str = '\n'.join(f'{k}: {v}' for k, v in info_dict.items())
    # ax.text(0.5, 0.5, info_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    '''


    # Apend the baselines for plotting
    new_data = [
        {'model': 'l0_quant', 'backbone_version': 'FP32 baseline', 'all': 78.509},
        {'model': 'l1_quant', 'backbone_version': 'FP32 baseline', 'all': 78.835},
        {'model': 'l2_quant', 'backbone_version': 'FP32 baseline', 'all': 79.13},
        {'model': 'xl0_quant', 'backbone_version': 'FP32 baseline', 'all': 79.752},
        {'model': 'xl1_quant', 'backbone_version': 'FP32 baseline', 'all': 79.93},
        {'model': 'l0_quant', 'backbone_version': 'INT8 baseline', 'all': 50.879099},
        {'model': 'l1_quant', 'backbone_version': 'INT8 baseline', 'all': 49.978849},
        {'model': 'l2_quant', 'backbone_version': 'INT8 baseline', 'all': 24.187811},
        {'model': 'xl0_quant', 'backbone_version': 'INT8 baseline', 'all': 50.059622},
        {'model': 'xl1_quant', 'backbone_version': 'INT8 baseline', 'all': 73.630043}
    ]
    new_data_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_data_df], ignore_index=True)

    # Rename models
    df['model'] = df['model'].str.replace('_quant', '')
    df['model'] = df['model'].str.upper()
    # Group the dataframe by 'backbone_version'
    grouped = df.groupby('backbone_version')
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


    for n in ("3","4","5"):
        fig, ax = plt.subplots()
        #iterate over the groups

        # TODO: Find any group whose name contains the string "baseline" and plot that group in the plot

        for name, group in grouped:
            if name.startswith(n):
                # Plot performance against base model for each backbone
                linestyle='solid' if 'only' in name else '--'
                ax.plot(group['model'], group['all'], label=name, linestyle=linestyle) # + np.random.uniform(0,0,len(group['all']))
            if 'baseline' in name:
                color = 'red' if 'FP32' in name else 'black'
                ax.plot(group['model'], group['all'], label=name, linestyle='dotted', marker = 'x', color=color)

        ax.set_xlabel('Base Model')
        ax.set_ylabel('box prompt "all" score')
        ax.set_title(f'Experiment 3, backbone benchmarking, family {n}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.savefig(f'./plots/plot_family_{n}.png', bbox_inches='tight')
        plt.close() 