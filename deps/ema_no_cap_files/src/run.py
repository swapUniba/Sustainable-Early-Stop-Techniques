import os
import time

# movielens_1m - amazon_books_60core
datasets = ['amazon_books_60core']
# ['BPR', 'DMF', 'LINE', 'MultiDAE', 'NGCF', 'DGCF', 'LightGCN', 'CKE', 'CFKG', 'KGCN', 'KGNNLS']
models = ['BPR', 'DMF', 'LINE', 'MultiDAE', 'NGCF', 'DGCF', 'LightGCN', 'CKE', 'CFKG', 'KGCN', 'KGNNLS']


tolerance_step  = 11
smoothing_factor = 2/(tolerance_step+1)

for dataset in datasets:
    for model in models:
        os.system(f"python src/tracker.py --dataset={dataset} --model={model} --tolerance_step={tolerance_step} --smoothing_factor={smoothing_factor}")
        time.sleep(30)