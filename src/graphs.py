import os
from config.global_config import get_global_config
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.scale
#from adjustText import adjust_text

_config = get_global_config()
BASE_PATH = os.getcwd()

# solo in locale (WIN10) - sublime || nel caso di problemi commenta riga sotto
PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))

# su Leonardo bastava il base_path e non tornare di sopra di una cartella ^
#PATH = BASE_PATH

BASE_PATH = os.path.join(PATH, _config.get('EXPERIMENT_RESULTS'))

SAVE_PATH = os.path.join(PATH, _config.get('GRAPH_PATH'))

models = ['BPR', 'DMF', 'LINE', 'MultiDAE', 'NGCF', 'DGCF', 'LightGCN', 'CKE', 'CFKG', 'KGCN', 'KGNNLS']
metrics_list = ['recall@10', 'ndcg@10','averagepopularity@10', 'giniindex@10']

classic_datasets_path = 'results_classic_early_stop'
datasets = ['movielens_1m', 'amazon_books_60core'] 

## Movielens 1M exp
ml1m_first_sol = [
    'results (13443369) [ml1m ~ 30 - 9]',
    'results (13445015) [ml1m ~ 30 - 11]',
    'results (13447229) [ml1m ~ 20 - 9]',
    'results (13448833) [ml1m ~ 20 - 11]',
    'results (13467479) [ml1m ~ 10 - 5]'
]

ml1m_second_sol_CAP = [
    'results (14219274) [ml1m ~ EMA - 15]',
    'results (13490790) [ml1m ~ EMA - 11]',
    'results (13486328) [ml1m ~ EMA - 9]',
    'results (13483044) [ml1m ~ EMA - 7]',
    'results (14206737) [ml1m ~ EMA - 5]'  
]

ml1m_second_sol_NO_CAP = [
    'results (14225361) [ml1m ~ EMA - 15 - NO CAP]',
    'results (13506803) [ml1m ~ EMA - 11 - NO CAP]',
    'results (13502110) [ml1m ~ EMA - 9 - NO CAP]',
    'results (13498346) [ml1m ~ EMA - 7 - NO CAP]',
    'results (14213752) [ml1m ~ EMA - 5 - NO CAP]'
]

## Amazon Books 60c exp
ab60c_first_sol = [
    'results (13934489) [ab60c ~ 10 - 5]',
    'results (13939913) [ab60c ~ 10 - 7]',
    'results (13944274) [ab60c ~ 5 - 7]',
    'results (13946879) [ab60c ~ 2 - 7]',
    'results (13950395) [ab60c ~ 2 - 9]'
]

ab60c_second_sol_CAP = [
    'results (14150827) [ab60c ~ EMA - 15]',
    'results (14233393) [ab60c ~ EMA - 11]',
    'results (13693133) [ab60c ~ EMA - 9]',
    'results (13685472) [ab60c ~ EMA - 7]',
    'results (13671274) [ab60c ~ EMA - 5]'
]

ab60c_second_sol_NO_CAP = [
    'results (14175867) [ab60c ~ EMA - 15 - NO CAP]',
    'results (14242840) [ab60c ~ EMA - 11 - NO CAP]',
    'results (13747310) [ab60c ~ EMA - 9 - NO CAP]',
    'results (13739772) [ab60c ~ EMA - 7 - NO CAP]',
    'results (13725523) [ab60c ~ EMA - 5 - NO CAP]'
]

# Def colori per modello
colors = {
    'BPR': '#1f77b4',      # Blu
    'DMF': '#ff7f0e',      # Arancione
    'NGCF': '#2ca02c',     # Verde
    'DGCF': '#d62728',     # Rosso
    'MultiDAE': '#9467bd', # Viola
    'LightGCN': '#8c564b', # Marrone
    'LINE': '#e377c2',     # Rosa
    'CKE': '#7f7f7f',      # Grigio
    'CFKG': '#bcbd22',     # Giallo oliva
    'KGCN': '#17becf',     # Turchese
    'KGNNLS': '#ff9896'    # Rosa chiaro
}

def plot_emission(path_dataset, classic_exp_path=None):

    if classic_exp_path is None:
        for dataset in datasets:
            emissions_result = {}
            dataset_exists = True
            for model in models:
                results_path = os.path.join(BASE_PATH, path_dataset, dataset, model)
                if not os.path.exists(results_path):
                    dataset_exists = False
                    break
                emissions = pd.read_csv(results_path + "/emissions.csv")
                emissions_result[model] = emissions.loc[0, 'emissions']

            if not dataset_exists:
                continue
            
            plt.figure(figsize=(8, 7))
            plt.yticks(fontsize=12)
            plt.ylabel("Emissions log2", fontsize=14)
            plt.xlabel("Models", fontsize=14)
            plt.xticks(rotation=55,fontsize=14)
            values=list(emissions_result.values())
            x_labels=list(emissions_result.keys())
            sorted_data = sorted(zip(values,x_labels), reverse=True)
            values, x_labels = zip(*sorted_data)
            plt.bar(x_labels, values, width=0.8, color='orange')
            for i, v in enumerate(values):
                plt.text(i, v + 0.000001, str(round(v, 3)), ha='center', va='bottom', fontsize=14)  
            plt.yscale(matplotlib.scale.LogScale(base=2, axis='y'))

            name_path = os.path.join(SAVE_PATH, path_dataset)

            if not os.path.exists(name_path):
                os.makedirs(name_path)
            
            name = dataset + " early classic"

            plt.title(f"{name}", fontsize=18)
            
            plt.tight_layout(pad=2.0)
            #plt.title(f"{dataset}", fontsize=18)
            plt.savefig(name_path+'/'+dataset+'_emissions.png')
            plt.close()
            del emissions_result, emissions
    else:
        # Caso con classic_exp_path (compara modelli con differenze colorate)
        for dataset in datasets:
            emissions_result = {}
            emissions_result_classic = {}

            dataset_exists = True
            for model in models:
                # Carica i risultati del nuovo modello (path_dataset)
                results_path = os.path.join(BASE_PATH, path_dataset, dataset, model)
                if not os.path.exists(results_path):
                    dataset_exists = False
                    break
                emissions = pd.read_csv(results_path + "/emissions.csv")
                emissions_result[model] = emissions.loc[0, 'emissions']

                # Carica i risultati del modello classico (classic_exp_path)
                results_path_classic = os.path.join(BASE_PATH, classic_exp_path, dataset, model)
                if not os.path.exists(results_path_classic):
                    dataset_exists = False
                    break
                emissions_classic = pd.read_csv(results_path_classic + "/emissions.csv")
                emissions_result_classic[model] = emissions_classic.loc[0, 'emissions']

            if not dataset_exists:
                continue

            plt.figure(figsize=(8, 7))
            plt.yticks(fontsize=12)
            plt.ylabel("Emissions log2", fontsize=14)
            plt.xlabel("Models", fontsize=14)
            plt.xticks(rotation=55, fontsize=14)

            # Ordinamento dei modelli per i risultati del modello
            values = list(emissions_result.values())
            x_labels = list(emissions_result.keys())
            sorted_data = sorted(zip(values, x_labels), reverse=True)
            values, x_labels = zip(*sorted_data)

            name = ""
            import re
            match = re.search(r'EMA\s*-\s*(\d+)', path_dataset)
            if match:
                name = "tolerance_step = " + match.group(1)+""
            else:
                match = re.search(r'~\s*(\d+)\s*-\s*(\d+)\s*\]', path_dataset)
                if match:
                    name = "(α - β): " + match.group(1) + " - " + match.group(2)

            # Sovrapponi le barre per i due set di dati
            for i, model in enumerate(x_labels):
                plt.bar(model, emissions_result_classic[model], width=0.8, color='orange', alpha=1.0, label='Early Classic' if i==0 else '')
                plt.bar(model, emissions_result[model], width=0.8, color='green', alpha=0.6, label=f'{name}' if i==0 else '')

                aux = 0.000001
                if emissions_result_classic[model] < emissions_result[model]:
                    aux = -1 * aux
                # Aggiungi etichetta sopra la barra per il modello classico (early classic)
                plt.text(i, emissions_result_classic[model] + aux, str(round(emissions_result_classic[model], 3)), ha='center', va='bottom' if aux > 0 else 'top', fontsize=14)

                # Aggiungi etichetta sotto la barra per il nuovo modello
                plt.text(i, emissions_result[model] - aux, str(round(emissions_result[model], 3)), ha='center', va='top' if aux > 0 else 'bottom', fontsize=14)

            plt.yscale(matplotlib.scale.LogScale(base=2, axis='y'))

            name_path = os.path.join(SAVE_PATH, path_dataset)

            if not os.path.exists(name_path):
                os.makedirs(name_path)

            plt.title(f"{dataset} ~ {name}", fontsize=18)

            # Aggiungi legenda
            plt.legend(fontsize=12)

            plt.tight_layout(pad=2.0)
            plt.savefig(name_path+'/'+dataset+'_emissions.png')
            plt.close()
            del emissions_result, emissions, emissions_result_classic, emissions_classic

def plot_metrics(path_dataset):
    for metric in metrics_list:
        for dataset in datasets:
            metrics_result={}
            emissions_result={}
            dataset_exists = True
            for model in models:
                results_path = os.path.join(BASE_PATH, path_dataset, dataset, model)
                if not os.path.exists(results_path):
                    dataset_exists = False
                    break
                metrics = pd.read_csv(results_path+"/metrics.csv")
                emissions= pd.read_csv(results_path+"/emissions.csv")
                metrics_result[model] = metrics[metric].iloc[0]
                emissions_result[model] = emissions['emissions'].iloc[-1]
            
            if not dataset_exists:
                continue

            plt.xlabel(metric, fontsize=14)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.ylabel("Emissions log2", fontsize=14)
            for i, model in enumerate(models):
                plt.plot(metrics_result[model], emissions_result[model], 'o', label=model, color='blue')
                plt.annotate(model, (metrics_result[model], emissions_result[model]), textcoords="offset points", xytext=(0,3), ha='center', fontsize=14)
            plt.yscale('log', base=2)
            plt.grid(True)

            name = dataset + " early classic"
            import re
            match = re.search(r'EMA\s*-\s*(\d+)', path_dataset)
            if match:
                name = dataset + " ~ tolerance_step = " + match.group(1)+""
            else:
                match = re.search(r'~\s*(\d+)\s*-\s*(\d+)\s*\]', path_dataset)
                if match:
                    name = dataset + " ~ (α - β): " + match.group(1) + " - " + match.group(2)
    
            plt.title(f"{name}", fontsize=18)

            name_path = os.path.join(SAVE_PATH, path_dataset)

            if not os.path.exists(name_path):
                os.makedirs(name_path)

            plt.savefig(name_path+'/'+dataset+'_'+metric+'.png')
            plt.close()
            del metrics_result, metrics

"""def compare_experiments(dataset_name, experiment_paths, experiment_name):
    for path in experiment_paths:
        if not os.path.exists(os.path.join(BASE_PATH, path, dataset_name)):
            print("Dataset non presente in tutti gli esperimenti da confrontare.")
            return

    for metric in metrics_list:
        plt.figure(figsize=(13, 7))
        plt.title(f"Comparison of {metric}", fontsize=14)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel("log2 Emissions", fontsize=12)

        metrics_results = {model: [] for model in models}
        emissions_results = {model: [] for model in models}

        for path in experiment_paths:
            for model in models:
                results_path = os.path.join(BASE_PATH, path, dataset_name, model)
                if os.path.exists(results_path):
                    metrics = pd.read_csv(os.path.join(results_path, "metrics.csv"))
                    emissions = pd.read_csv(os.path.join(results_path, "emissions.csv"))

                    metrics_results[model].append(metrics[metric].iloc[0])
                    emissions_results[model].append(emissions['emissions'].iloc[-1])

        dataset_aliases_sequence = [os.path.basename(path) for path in experiment_paths]

        texts = []
        added_legends = set()

        for model in models:
            if model in metrics_results:
                x_vals = metrics_results[model]
                y_vals = emissions_results[model]

                for i in range(len(experiment_paths)):
                    color = colors.get(model, '#000000')
                    label = dataset_aliases_sequence[i] if dataset_aliases_sequence[i] not in added_legends else ""
                    plt.scatter(x_vals[i], y_vals[i], color=color, label=label)

                    if i == 0:
                        texts.append(plt.text(x_vals[i], y_vals[i], model, fontsize=10))

                    added_legends.add(dataset_aliases_sequence[i])

                for i in range(len(experiment_paths) - 1):
                    x1 = x_vals[i]
                    y1 = y_vals[i]
                    x2 = x_vals[i + 1]
                    y2 = y_vals[i + 1]

                    plt.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)

        #adjust_text(texts)
        plt.grid(True)
        plt.yscale('log', base=2)
        plt.legend(loc='best')

        name_path = os.path.join(SAVE_PATH, experiment_name)

        if not os.path.exists(name_path):
            os.makedirs(name_path)
        
        #experiment_save_path = os.path.join(SAVE_PATH, f'{experiment_name}_{metric}_comparison.png')
        plt.savefig(name_path+'/'+'comparison_'+metric+'.png')
        plt.close()"""

def compare_experiments(dataset_name, mod_exp_paths, experiment_name):

    experiments_modified_list = []
    # Caricamento dei dati mod stopping
    for path in mod_exp_paths:
        metrics_result2 = {}
        emissions_result2 = {}
        for model in models:
            results_path = os.path.join(BASE_PATH, path, dataset_name, model)
            metrics = pd.read_csv(os.path.join(results_path, "metrics.csv"))
            emissions = pd.read_csv(os.path.join(results_path, "emissions.csv"))
            metrics_result2[model] = metrics
            emissions_result2[model] = emissions
        experiments_modified_list.append([metrics_result2, emissions_result2])
    
    # Plot
    for metric in metrics_list:
        for model in models:
            emission_difference = []
            performance_difference = []
            texts = [] ##
            model_points = [] ##
            for i, experiment in enumerate(experiments_modified_list):
                emission_diff = experiment[1][model].loc[0, 'emissions'] * 1000
                performance_diff = experiment[0][model][metric].iloc[-1]
                emission_difference.append(emission_diff)
                performance_difference.append(performance_diff)

                model_points.append((performance_diff, emission_diff)) ##

                import re
                match = re.search(r'EMA\s*-\s*(\d+)', mod_exp_paths[i])
                if match:
                    extracted_part = match.group(1)
                else:
                    match = re.search(r'~\s*(\d+)\s*-\s*(\d+)\s*\]', mod_exp_paths[i])
                    if match:
                        extracted_part = match.group(1) + " - " + match.group(2)
                    else:
                        extracted_part = "▼"

                plt.annotate(extracted_part, (performance_diff, emission_diff), fontsize=15, textcoords="offset points", xytext=(0, 3))
                #text = plt.text(performance_diff, emission_diff, extracted_part, fontsize=10, color=colors.get('#000000'))
                #texts.append(text)
            
            ##
            performance_diff_values, emission_diff_values = zip(*model_points)
            plt.plot(performance_diff_values, emission_diff_values, color=colors.get(model, '#000000'), linewidth=0.85, linestyle='-', alpha=0.7)

            plt.scatter(performance_difference, emission_difference, label=model, color=colors.get(model, '#000000'))
            #adjust_text(texts)
        
        plt.scatter([], [], label="early classic", color='black', marker='v', s=100, edgecolor='black')
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.ylabel('log2 Emission (g)', fontsize=16)
        plt.xlabel(f'{metric}', fontsize=16)
        plt.title(f'Comparison of {metric}', fontsize=18)
        plt.gcf().set_size_inches(20, 12)
        plt.grid(True)
        plt.tight_layout()
        plt.yscale('log', base=2)
        plt.legend(title="Models", loc='best', fontsize=14)

        name_path = os.path.join(SAVE_PATH, experiment_name)

        if not os.path.exists(name_path):
            os.makedirs(name_path)

        #plt.savefig(os.path.join(SAVE_PATH, f'sensibility_{metric}_comparison.png'))
        plt.savefig(name_path+'/'+'comparison_'+metric+'.png')
        plt.close()


def sensibilityGraph(dataset_name, mod_exp_paths, classic_exp_path, experiment_name):
    emissions_result = {}
    metrics_result = {}
    experiments_modified_list = []

    # Caricamento early classic
    for model in models:
        results_path = os.path.join(BASE_PATH, classic_exp_path, dataset_name, model)
        metrics = pd.read_csv(os.path.join(results_path, "metrics.csv"))
        emissions = pd.read_csv(os.path.join(results_path, "emissions.csv"))
        metrics_result[model] = metrics
        emissions_result[model] = emissions

    # Caricamento dei dati mod stopping
    for path in mod_exp_paths:
        metrics_result2 = {}
        emissions_result2 = {}
        for model in models:
            results_path = os.path.join(BASE_PATH, path, dataset_name, model)
            metrics = pd.read_csv(os.path.join(results_path, "metrics.csv"))
            emissions = pd.read_csv(os.path.join(results_path, "emissions.csv"))
            metrics_result2[model] = metrics
            emissions_result2[model] = emissions
        experiments_modified_list.append([metrics_result2, emissions_result2])

    # Plot
    for metric in metrics_list:
        for model in models:
            emission_difference = []
            performance_difference = []
            texts = [] ##
            model_points = [] ##
            for i, experiment in enumerate(experiments_modified_list):
                emission_diff = abs(emissions_result[model].loc[0, 'emissions'] - experiment[1][model].loc[0, 'emissions']) * 1000
                performance_diff = abs(metrics_result[model][metric].iloc[-1] - experiment[0][model][metric].iloc[-1])
                emission_difference.append(emission_diff)
                performance_difference.append(performance_diff)

                model_points.append((performance_diff, emission_diff)) ##

                import re
                match = re.search(r'EMA\s*-\s*(\d+)', mod_exp_paths[i])
                if match:
                    extracted_part = match.group(1)
                else:
                    match = re.search(r'~\s*(\d+)\s*-\s*(\d+)\s*\]', mod_exp_paths[i])
                    if match:
                        extracted_part = match.group(1) + " - " + match.group(2)

                plt.annotate(extracted_part, (performance_diff, emission_diff), fontsize=14, textcoords="offset points", xytext=(2, 2))
                #text = plt.text(performance_diff, emission_diff, extracted_part, fontsize=10, color=colors.get('#000000'))
                #texts.append(text)
            
            ##
            performance_diff_values, emission_diff_values = zip(*model_points)
            plt.plot(performance_diff_values, emission_diff_values, color=colors.get(model, '#000000'), linewidth=0.5, linestyle='-', alpha=0.7)

            plt.scatter(performance_difference, emission_difference, label=model, color=colors.get(model, '#000000'))
            #adjust_text(texts)

        plt.ylabel('log2 ABS Emission decrease (g)', fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel(f'{metric} ABS decrease', fontsize=14)
        plt.title(f'Performance decrease vs Emission decrease - {metric}', fontsize=18)
        plt.gcf().set_size_inches(16, 8)
        plt.grid(True)
        plt.yscale('log', base=2)
        plt.legend(title="Models", loc='best', fontsize=14)

        name_path = os.path.join(SAVE_PATH, experiment_name)

        if not os.path.exists(name_path):
            os.makedirs(name_path)

        #plt.savefig(os.path.join(SAVE_PATH, f'sensibility_{metric}_comparison.png'))
        plt.savefig(name_path+'/'+'sensibilityGraph_'+metric+'.png')
        plt.close()

def print_table(dataset_name, mod_exp_paths, classic_exp_path):
    
    #name_path = os.path.join(SAVE_PATH, experiment_name)
    #out_file = name_path+'/'+name_path+"_comparison.txt"

    emissions_result = {}
    metrics_result = {}
    experiments_modified_list = []

    # Caricamento early classic
    for model in models:
        results_path = os.path.join(BASE_PATH, classic_exp_path, dataset_name, model)
        metrics = pd.read_csv(os.path.join(results_path, "metrics.csv"))
        emissions = pd.read_csv(os.path.join(results_path, "emissions.csv"))
        metrics_result[model] = metrics
        emissions_result[model] = emissions

    # Caricamento dei dati mod stopping
    for path in mod_exp_paths:
        metrics_result2 = {}
        emissions_result2 = {}
        for model in models:
            results_path = os.path.join(BASE_PATH, path, dataset_name, model)
            metrics = pd.read_csv(os.path.join(results_path, "metrics.csv"))
            emissions = pd.read_csv(os.path.join(results_path, "emissions.csv"))
            metrics_result2[model] = metrics
            emissions_result2[model] = emissions
        experiments_modified_list.append([metrics_result2, emissions_result2])

    # Plot
    for metric in metrics_list:
        for model in models:
            print("\n\n >> ", model, " - ", metric)
            for i, experiment in enumerate(experiments_modified_list):
                name = "error"
                import re
                match = re.search(r'EMA\s*-\s*(\d+)', mod_exp_paths[i])
                if match:
                    name =  match.group(1)
                else:
                    match = re.search(r'~\s*(\d+)\s*-\s*(\d+)\s*\]', mod_exp_paths[i])
                    if match:
                        name = match.group(1) + " - " + match.group(2)
                
                metric_classic = metrics_result[model][metric].iloc[-1]
                metric_mod = experiment[0][model][metric].iloc[-1]

                emission_classic = emissions_result[model].loc[0, 'emissions']
                emission_mod = experiment[1][model].loc[0, 'emissions']
                
                _metric = round(metric_mod,4)
                metric_red = round((metric_mod - metric_classic) / metric_classic * 100, 2)
                diff_metric = round((metric_mod - metric_classic),4)

                _emission = round(emission_mod,6)
                emission_red = round((emission_mod - emission_classic) / emission_classic * 100, 2)
                diff_emission= round((emission_mod - emission_classic),6)

                aux_1 = " "
                if metric_red > 0:
                    aux_1 = " +"

                aux_2 = " "
                if emission_red > 0:
                    aux_2 = " +"

                print(f"{name} & {_metric} &{aux_1}{metric_red} &{aux_1}{diff_metric} & {_emission} &{aux_2}{emission_red} &{aux_2}{diff_emission} \\\\")

# -------------------------------------------------------------------------------------------------------------------------------------------- #

experiments = (
    ml1m_first_sol +
    ml1m_second_sol_CAP +
    ml1m_second_sol_NO_CAP +
    ab60c_first_sol +
    ab60c_second_sol_CAP +
    ab60c_second_sol_NO_CAP
)

plot_emission(classic_datasets_path)
plot_metrics(classic_datasets_path)

for experiment_path in experiments:
    if os.path.exists(os.path.join(BASE_PATH, experiment_path)):
        plot_emission(experiment_path, classic_datasets_path)
        plot_metrics(experiment_path)


#ml1m
compare_experiments('movielens_1m', ml1m_first_sol + [classic_datasets_path], 'ml1m_first_sol')
compare_experiments('movielens_1m', ml1m_second_sol_CAP + [classic_datasets_path], 'ml1m_second_sol_CAP')
compare_experiments('movielens_1m', ml1m_second_sol_NO_CAP + [classic_datasets_path], 'ml1m_second_sol_NO_CAP')

#ab60c
compare_experiments('amazon_books_60core', ab60c_first_sol + [classic_datasets_path], 'ab60c_first_sol')
compare_experiments('amazon_books_60core', ab60c_second_sol_CAP + [classic_datasets_path], 'ab60c_second_sol_CAP')
compare_experiments('amazon_books_60core', ab60c_second_sol_NO_CAP + [classic_datasets_path], 'ab60c_second_sol_NO_CAP')


metrics_list = ['recall@10', 'giniindex@10', ] # 'giniindex@10' - 'recall@10'
# nome gruppo dell'esperimento da analizzare
name_exp = ab60c_second_sol_NO_CAP
print_table('amazon_books_60core', name_exp, classic_datasets_path)