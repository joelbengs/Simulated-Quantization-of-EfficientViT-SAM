import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt 

def read_pickle_to_dataframe(file_path, file_name) -> pd.DataFrame:
        file_name = os.path.basename(file_name)
        file_name = os.path.splitext(file_name)[0]
        file_string = f'{file_path}/{file_name}.pkl'

        with open(file_string, 'rb') as file: # read binary
            data = pickle.load(file)
        df = pd.DataFrame(data)
        return df, file_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read pickle file and convert it to pandas dataframe')
    parser.add_argument('--pickle_file_path', type=str, default='results_storage', help='The directory of the pickle file')
    parser.add_argument('--pickle_file_name', type=str, help='The name of the pickle file to analyse')
    args = parser.parse_args()

    # load the data from storage
    df, _ = read_pickle_to_dataframe(args.pickle_file_path, args.pickle_file_name)

    # Print data overview
    selected_columns = ['model',
                        'backbone_version',
                        'all']
    intersection_columns = [col for col in selected_columns if col in df.columns]
    selected_data = df[intersection_columns]
    print("data analytics script is working with the following data:")
    print(selected_data)

    # Apend the baselines for plotting - will need to extract them later.
    '''new_data = [
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
'''
    # Rename models to remove _quant ending
    df['model'] = df['model'].str.replace('_quant', '')
    df['model'] = df['model'].str.upper()
    # Group the dataframe by 'backbone_version'
    grouped = df.groupby('backbone_version')
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


    for n in ("3","4","5_q_all", "5_q_only"):
        fig, ax = plt.subplots()
        #iterate over the groups
        for i, (name, group) in enumerate(grouped):
            if name.startswith(n):
                # Plot performance against base model for each backbone
                linestyle='dotted' if i % 2 == 0 else '--'
                ax.plot(group['model'], group['all'], label=name, linestyle=linestyle) # + np.random.uniform(0,0,len(group['all']))
            elif 'baseline' in name:
                color = 'red' if 'FP32' in name else 'black'
                ax.plot(group['model'], group['all'], label=name, linestyle='dotted', marker = 'x', color=color)

        ax.set_xlabel('Base Model')
        ax.set_ylabel('box prompt "all" score')
        ax.set_title(f'Experiment 3V2, backbone benchmarking, family {n}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.savefig(f'./plots/plot_family_{n}_v2.png', bbox_inches='tight')
        plt.close() 







    # Get metadata for the textbox
    '''
    first_row = df.iloc[0]
    meta_columns = [col for col in df.columns if col not in selected_columns]     
    info_dict = {col: first_row[col] for col in meta_columns}
    info_str = '\n'.join(f'{k}: {v}' for k, v in info_dict.items())
    # ax.text(0.5, 0.5, info_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    '''

    base_model_sizes = {
        'L0': {'Total params': 30728224, 'Total mult-adds (G)': 104.45},
        'L1': {'Total params': 43585568, 'Total mult-adds (G)': 128.74},
        'L2': {'Total params': 57264032, 'Total mult-adds (G)': 174.16},
        'XL0': {'Total params': 112893344, 'Total mult-adds (G)': 182.95},
        'XL1': {'Total params': 199281568, 'Total mult-adds (G)': 318.81}
        }


    model_params_per_stage = {
        'L0': {
            'Total params': 30728224,
            'Total mult-adds (G)': 104.45,
            'stage0': 19488,
            'stage1': 345856,
            'stage2': 1379840,
            'stage3': 2953728,
            'stage4': 23106560,
            'stage5': 0,
            'stage0_bottleneck': 928,
            'stage1_bottleneck': 181376,
            'stage2_bottleneck': 723200,
            'stage3_bottleneck': 809472,
            'stage4_bottleneck': 4787200,
            'stage5_bottleneck': 0,
        },
        'L1': {
            'Total params': 43585568, 
            'Total mult-adds (G)': 128.74,
            'stage0': 19488,
            'stage1': 345856,
            'stage2': 1379840,
            'stage3': 4025856,
            'stage4': 32266240,
            'stage5': 0,
            'stage0_bottleneck': 928,
            'stage1_bottleneck': 181376,
            'stage2_bottleneck': 723200,
            'stage3_bottleneck': 809472,
            'stage4_bottleneck': 4787200,
            'stage5_bottleneck': 0,
        },
        'L2': {
            'Total params': 57264032, 
            'Total mult-adds (G)': 174.16,
            'stage0': 19488,
            'stage1': 510336,
            'stage2': 2036480,
            'stage3': 5097984,
            'stage4': 41425920,
            'stage5': 0,
            'stage0_bottleneck': 928,
            'stage1_bottleneck': 181376,
            'stage2_bottleneck': 723200,
            'stage3_bottleneck': 809472,
            'stage4_bottleneck': 4787200,
            'stage5_bottleneck': 0,
        },
        'XL0': {
            'Total params': 112893344, 
            'Total mult-adds (G)': 182.95,
            'stage0': 928,
            'stage1': 345856,
            'stage2': 1379840,
            'stage3': 8136192,
            'stage4': 13678080,
            'stage5': 73081856,
            'stage0_bottleneck': 928,
            'stage1_bottleneck': 181376,
            'stage2_bottleneck': 723200,
            'stage3_bottleneck': 2888192,
            'stage4_bottleneck': 3191808,
            'stage5_bottleneck': 19011584,
        },
        'XL1': {
            'Total params': 199281568, 
            'Total mult-adds (G)': 318.81,
            'stage0': 19488,
            'stage1': 510336,
            'stage2': 2036480,
            'stage3': 13384192,
            'stage4': 24164352,
            'stage5': 127152128,
            'stage0_bottleneck': 928,
            'stage1_bottleneck': 181376,
            'stage2_bottleneck': 723200,
            'stage3_bottleneck': 2888192,
            'stage4_bottleneck': 3191808,
            'stage5_bottleneck': 19011584,
        }
    }

