import os
import time

datasets = ['amazon_books_60core'. 'movielens_1m']

models = ['BPR', 'DMF', 'LINE', 'MultiDAE', 'NGCF', 'DGCF', 'LightGCN', 'CKE', 'CFKG', 'KGCN', 'KGNNLS']

trade_off = 2
max_emission_step = 9
# smoothing_factor = 2/(max_emission_step+1)

for dataset in datasets:
    for model in models:
        os.system(f"python src/tracker.py --dataset={dataset} --model={model} --max_emission_step={max_emission_step} --trade_off={trade_off}")
        #os.system(f"python src/tracker.py --dataset={dataset} --model={model} --max_emission_step={max_emission_step} --smoothing_factor={smoothing_factor}")
        time.sleep(30)