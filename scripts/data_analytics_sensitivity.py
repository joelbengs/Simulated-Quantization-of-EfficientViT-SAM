# Author: Joel Bengs. Apache-2.0 license
# This code produces graphs from the quantization experiments on layer-wise sensitivity.

import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from efficientvit.quant_backbones_zoo import REGISTERED_BACKBONE_DESCRIPTIONS_LARGE, REGISTERED_BACKBONE_DESCRIPTIONS_XL, REGISTERED_BACKBONE_COLORS
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def read_pickle_to_dataframe(file_path, file_name) -> pd.DataFrame:
        file_name = os.path.basename(file_name)
        file_name = os.path.splitext(file_name)[0]
        file_string = f'{file_path}/{file_name}.pkl'

        with open(file_string, 'rb') as file: # read binary
            data = pickle.load(file)
        df = pd.DataFrame(data)
        return df, file_string

''' Function that plots blockwise and layerwise benchmarking '''
def plot(df, title: str, xlabel: str, name: str, model: str = 'L0', prompt_type: str = 'box', add_descriptions=True, rotate=False, zoom=False):
    baselines_box = { # measured empirically
                'L0':78.509,
                'L1':78.835,
                'L2':79.130,
                'XL0':79.752,
                'XL1':79.930,
            }
    
    if model.startswith('X'):
        description_dict = REGISTERED_BACKBONE_DESCRIPTIONS_XL
    else:
        description_dict = REGISTERED_BACKBONE_DESCRIPTIONS_LARGE
    performance_measure = 'all'
    baseline_value=baselines_box[model]
    upper_limit=85
    zoom_lower_limit=76.9
    zoom_upper_limit = baseline_value + 0.2
    fig, ax = plt.subplots(figsize=(25,4))

    ax.plot(df['backbone_version'], df[performance_measure], color='slategrey')

    for i, val in df.iterrows(): # index and row data
        ax.scatter(
                    val['backbone_version'], 
                    val[performance_measure],
                    color=REGISTERED_BACKBONE_COLORS.get(description_dict[val['backbone_version']], 'grey'),
                    s=100)
        # vertical lines at each x mark
        ax.axvline(val['backbone_version'], color='lightgray', linestyle='dotted')

    ax.axhline(baseline_value, color='black', linestyle='dotted')
    ax.axhline(zoom_lower_limit + 0.1, color='darkgreen', linestyle='dashdot')

    if not zoom:
        color_dict = {
            'Attention' : 'darkturquoise',
            'MBC' : 'orangered',
            'Fused-MBC' : 'limegreen',
            'ResBlock': 'fuchsia',
            'Other': 'grey',
        }

        # Create a Line2D object for the black dotted line
        baseline_line = mlines.Line2D([], [], color='black', linestyle='dotted', label='Baseline')
        zoom_line = mlines.Line2D([], [], color='darkgreen', linestyle='dashdot', label='Zoom guideline')
        legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
        legend_handles.append(baseline_line)
        legend_handles.append(zoom_line)
        ax.legend(handles=legend_handles, loc='lower right', fontsize=14)

    if zoom: ax.set_xlabel(xlabel, fontsize=14)
    if not zoom:
        plt.suptitle(title, fontsize=14)

    ax.tick_params(axis='x', labelsize=10) # to control size of x-ticks
    ax.set_ylabel('mIoU (higher is better)', fontsize=14)
    if rotate:
        plt.xticks(rotation=90)
    if zoom:
        ax.set_ylim([zoom_lower_limit,zoom_upper_limit])
        save_name = f'./plots/graphs/{name}_color_zoom.png'
    else:
        ax = plt.gca()  # get current axes
        ymin, _ = ax.get_ylim()  # get the current lower limit
        ax.set_ylim(ymin, upper_limit)  # set the new upper limit
        save_name = f'./plots/graphs/{name}_color.png'
    
    # get rid of the whitespace on either side
    sorted_versions = sorted(df['backbone_version'])
    ax.set_xlim(sorted_versions[0], sorted_versions[-1])

    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='From pickle file to plots')
    parser.add_argument('--file_path', type=str, default='results', help='The directory of the pickle file')
    parser.add_argument('--file_name', type=str, help='The name of the pickle file to analyse')
    parser.add_argument('--model', type=str, default='L0',  help='The model of the test, in capital letters')
    args = parser.parse_args()

    # load the data from storage
    df, _ = read_pickle_to_dataframe(args.file_path, args.file_name)

    pd.set_option('display.max_rows', None)
    print("data analytics script is working with the following data:")
    print(df.columns)

    ################################################################
    ###                      CONFIGURATION                       ###
    ###   Experiment - simualted layer activation quantization   ###
    ################################################################

    # Config for plot's text:
    # model = 'L0' # 'L1', 'L2', 'XL0', XL1'
    model = args.model
    quant_scheme = 'simulated integer-only (weights + activations)' # 'simulated weight-only'
    prompt_type = 'box' #or 'point'

    ################################################################
    ###                      CONFIGURATION                       ###
    ###   Experiment - simualted layer activation quantization   ###
    ################################################################


    df_layer = df.copy()

    df_layer['model'] = df_layer['model'].str.replace('_quant', '')
    df_layer['model'] = df_layer['model'].str.upper()

    df_layer['backbone_version'] = df_layer['backbone_version'].str.replace(model + ':', '')
    df_layer['backbone_version'] = df_layer['backbone_version'].str.replace('stage', '')

    # split data: stage:block:x vs stage:block:layer
    df_layer = df_layer[~df_layer['backbone_version'].str.endswith('x')] # remove blockwise experiments, if any
    

    # plot normal
    plot(
        df_layer,
        title=f'Layer-wise sensitivity of {model}',
        xlabel=f'EfficientViT-SAM-{model}\'s image encoder with only one layer quantized using simulated quantization. Naming scheme is stage:block:layer.',
        name=f'{args.file_name}_layer',
        model = model,
        prompt_type=prompt_type,
        add_descriptions=True,
        rotate=True,
        zoom=False,
        )
    
    # plot zoomed
    plot(
        df_layer,
        title=f'Layer-wise sensitivity of {model}',
        xlabel=f'EfficientViT-SAM-{model}\'s image encoder with only one layer quantized using simulated quantization. Naming scheme is stage:block:layer.',
        name=f'{args.file_name}_layer',
        model = model,
        prompt_type=prompt_type,
        add_descriptions=True,
        rotate=True,
        zoom=True,
        )
    