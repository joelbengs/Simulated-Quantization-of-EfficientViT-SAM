import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playground')
    parser.add_argument('--export', action='store_true')

    args = parser.parse_args()

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

df = pd.DataFrame(model_params_per_stage).T

for i in range(6):
    df[f'stage{i}_bulk'] = df[f'stage{i}'] - df[f'stage{i}_bottleneck']


# add thousands seperators and convert to int
'''for col in df.columns:
    if col != 'Total mult-adds (G)':
        df[col] = df[col].apply(lambda x: "{:,}".format(x))'''

'''
def millions(x):
    return str(round(x / 1_000_000)) + 'M'

for col in df.columns:
    if col != 'Total mult-adds (G)':
        df[col] = df[col].apply(millions)
'''

def convert(x):
    if x > 1_000_000:
        return str(round(x / 1_000_000)) + 'M'
    else:
        return str(round(x / 1_000)) + 'K'
 

for col in df.columns:
    if col != 'Total mult-adds (G)':
        df[col] = df[col].apply(convert)



print(df)

print(df.head())


if args.export:
    excel_file_string = 'results/model_sizes.xlsx'
    df.to_excel(excel_file_string, sheet_name='parameters')
    print(f"Exported parameters to excel file named {excel_file_string}")