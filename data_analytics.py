import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from configs.quant_backbones_zoo import REGISTERED_BACKBONE_DESCRIPTIONS

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

    ################################
    ###       Experiment 5       ###
    ################################

    # Rename models to remove _quant ending
    df['model'] = df['model'].str.replace('_quant', '')
    df['model'] = df['model'].str.upper()
    df['backbone_version'] = df['backbone_version'].str.replace('L0:', '')

    fig, ax = plt.subplots()
    ax.plot(df['backbone_version'], df['all'])

    ax.set_xlabel('layer quantized')
    ax.set_ylabel('box prompt "all" score')
    ax.set_ylim([45,85])
    ax.xticks(rotation=90)
    
    plt.suptitle(f'Experiment 5, layer-wise analysis')
    plt.savefig(f'./plots/E5.png', bbox_inches='tight')
    plt.close() 


    ################################
    ###       Experiment 4       ###
    ################################
    '''
    #Add negative noise to XL1 make sure all models are visible in plot
    #mask = df['model'] == 'xl0_quant'
    #df.loc[mask,'all'] -= np.random.uniform(0, 2, size=df[mask].shape[0])

    # Rename models to remove _quant ending
     
    df['model'] = df['model'].str.replace('_quant', '')
    df['model'] = df['model'].str.upper()

    df = df[df['model'] != 'L2']
    df = df[df['model'] != 'XL0']

    # Group the dataframe by 'backbone_version'
    grouped = df.groupby('backbone_version')
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for n in ("3","4","5", "6", "7", "8","9"):
        fig, ax = plt.subplots()
        #iterate over the groups
        for i, (name, group) in enumerate(grouped):
            if name.startswith(n):
                # Plot performance against base model for each backbone
                linestyle='dotted' if i % 2 == 0 else '--'
                label=f"{name}\n(values: {' '.join([str(round(val, 1)) for val in group['all'].values])})"
                ax.plot(group['model'], group['all'], label=label, linestyle=linestyle) # + np.random.uniform(0,0,len(group['all']))
            elif 'baseline' in name:
                color = 'red' if 'FP32' in name else 'black'
                label=f"{name}\n(values: {' '.join([str(round(val, 1)) for val in group['all'].values])})"
                ax.plot(group['model'], group['all'], label=label, linestyle='dotted', marker = 'x', color=color)

        ax.set_xlabel('Base Model')
        ax.set_ylabel('box prompt "all" score')
        ax.set_ylim([0,85])
        
        ax.set_title(REGISTERED_BACKBONE_DESCRIPTIONS[str(n)])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle(f'Experiment 4, backbone benchmarking, family {n}')
        plt.savefig(f'./plots/E4_plot_family_{n}_L2_XL0_removed.png', bbox_inches='tight')
        plt.close() 


    # MODEL SIZE PLOT
    df = df[df['model'] != 'L1']
    grouped = df.groupby('backbone_version')

    for n in ("3","4","5", "6", "7", "8","9"):
        fig, ax = plt.subplots()
        #iterate over the groups
        for i, (name, group) in enumerate(grouped):
            if name.startswith(n):
                # Plot performance against base model for each backbone
                linestyle='dotted'
                label=f"{name}\n(values: {' '.join([str(round(val, 1)) for val in group['all'].values])})"
                ax.plot(group['model_size_mb_quantized'], group['all'], label=label, linestyle=linestyle, marker = 'o') # + np.random.uniform(0,0,len(group['all']))
            elif 'baseline' in name:
                color = 'red' if 'FP32' in name else 'black'
                label=f"{name}\n(values: {' '.join([str(round(val, 1)) for val in group['all'].values])})"
                ax.plot(group['model_size_mb_quantized'], group['all'], label=label, linestyle='-.', marker = 'x', color=color)

        ax.set_xlabel('theorethical model size in megabytes (lower is better)\n Left datapoint is LO as base model, right datapoint is XL0')
        ax.set_ylabel('box prompt "all" score (higher is better)')
        ax.set_ylim([40,85])
        
        ax.set_title(REGISTERED_BACKBONE_DESCRIPTIONS[str(n)])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle(f'Family {n}, L0 and XL1 only')
        plt.subplots_adjust(top=0.85)
        plt.savefig(f'./plots/E4_sizeplot_family_{n}.png', bbox_inches='tight')
        plt.close() 

    '''

    # Get metadata for the textbox
    '''
    first_row = df.iloc[0]
    meta_columns = [col for col in df.columns if col not in selected_columns]     
    info_dict = {col: first_row[col] for col in meta_columns}
    info_str = '\n'.join(f'{k}: {v}' for k, v in info_dict.items())
    # ax.text(0.5, 0.5, info_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    '''

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

