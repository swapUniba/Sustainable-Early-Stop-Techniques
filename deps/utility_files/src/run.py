import os
import time

# movielens_1m - amazon_books_60core
datasets = ['amazon_books_60core']
# ['BPR', 'DMF', 'LINE', 'MultiDAE', 'NGCF', 'DGCF', 'LightGCN', 'CKE', 'CFKG', 'KGCN', 'KGNNLS']
models = ['BPR', 'DMF', 'LINE', 'MultiDAE', 'NGCF', 'DGCF', 'LightGCN', 'CKE', 'CFKG', 'KGCN', 'KGNNLS']

trade_off = 2
max_emission_step = 9

for dataset in datasets:
    for model in models:
        os.system(f"python src/tracker.py --dataset={dataset} --model={model} --max_emission_step={max_emission_step} --trade_off={trade_off}")
        time.sleep(30)