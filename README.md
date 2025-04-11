# ♻️ Sustainable-Early-Stop-Techniques

Questa tesi esplora nuove strategie di early stopping per ridurre l'impatto ambientale (e dunque le emissioni di CO2) durante l'addestramento di modelli di raccomandazione, senza cercare di compromettere in modo sostanziale le performance (rispetto al metodo "classico" di early stopping).

Il lavoro è ereditato dalla seguente repository: [sustainability-of-recsys](https://github.com/albertovalerio/sustainability-of-recsys)
<br><br>

---

## 🎯 Obiettivo
L’obiettivo principale di questo lavoro è analizzare e migliorare il trade-off tra emissioni e performance nei sistemi di raccomandazione, concentrandosi sulla tecnica dell’early stopping.
<br><br>

---

## 🔧 Funzionamento
Il sistema tiene traccia delle emissioni di un determinato algoritmo di raccomandazione e su un determinato dataset.<br>
Esegue il modello di raccomandazione applicando il set di iperparametri predefiniti (o in alternativa si potrebbe eseguire una Grid Search per fare il tuning degli iperparametri grazie allo script ```src/tuning_tracker.py```, più informazioni sono presenti nella [Repo citata](https://github.com/albertovalerio/sustainability-of-recsys)).<br>
Vengono salvate durante l'esecuzione di ogni modello: le metriche, le emissioni e la configurazione dei parametri usati.

I modelli, i dataset e le metriche fanno riferimento all'implementazione di [@Recbole](https://recbole.io/).<br>
Il tracciamento delle emissioni è stato reso possibile attraverso la libreria [@CodeCarbon](https://mlco2.github.io/codecarbon/).
<br><br>

---

## 🛠️ Requisiti
Ambiente di sviluppo usato in fase sperimentale:<br>
```python/3.10.8--gcc--8.5.0```  + (opzionale) ```cuda/12.1```

Requisiti di sistema: [requirements.txt](https://github.com/Vincy02/Sustainable-Early-Stop-Techniques/blob/main/requirements.txt)
<br><br>

---

## 📁 Struttura del progetto
```
├── data/                   # Sono contenuti i dataset usati
├── deps/                   # Directory contenente dipendenze specifiche
│   ├── __other/            # Contiene script utili per visualizzare funzionamento delle tecniche sviluppate
│   ├── codecarbon/         # File e configurazioni relativi alla libreria CodeCarbon
│   ├── ema_files/          # File inerenti alla strategia di early stopping EMA (con limite superiore)
│   ├── ema_no_cap_files/   # File inerenti alla strategia di early stopping EMA (SENZA limite superiore)
│   └── utility_files/      # File inerenti alla strategia di early stopping Utility
├── experiment_results/     # Directory per i risultati dettagliati di ciascun esperimento
├── graphs/                 # Contiene tutti i grafici relativi ai risultati ottenuti
├── log/                    # Directory per i file di log relativi all'esecuzione degli script.
├── log_tensorboard/        # Directory per i log di TensorBoard
├── results/                # Directory principale per il salvataggio dei risultati
├── saved/                  # Directory per salvare i modelli addestrati
├── src/                    # Contiene il codice sorgente del progetto
│   ├── . . .
│   ├── graphs.py           # Script Python per generare i grafici presenti nella cartella "graphs/"
│   ├── ema_no_cap_files/   # File inerenti alla strategia di early stopping EMA (SENZA limite superiore)
│   └── utility_files/      # File inerenti alla strategia di early stopping Utility
├── requirements.txt        # Elenco delle dipendenze Python
└── README.md
```

Spostare la cartella ```codecarbon/``` presente nella cartella ```deps/``` nella cartella delle dipendenze dell'ambiente di lavoro.

Selezionare una delle strategie di Early Stopping sviluppata, presente nalla cartella ```deps/``` [*ema_files/*, *ema_no_cap_files/*, *utility_files/*] e spostare i file contenuti della cartella della strategia nelle cartelle corrispondenti:
* ```deps/<nome_strategia>/recbole``` nelle dipendenze dell'ambiente di lavoro
<br><br>
* ```deps/<nome_strategia>/src``` nella cartella root del progetto [verranno sostituiti dunque i file ```run.py``` e ```tracker.py```]<br>

Di default nella cartella ```src/``` presente nella root del progetto sono presenti i file ```run.py``` e ```tracker.py``` inerenti alla strategia **Utility**.

---

Per eseguire l'intero programma è possibile runnare direttamente il ```run.py``` presente nella cartella ```src```:
```python
$ python src/run.py 
```

Oppure se si volesse eseguire un modello/i e dataset in particolare (specificando debitamente i paramentri della metodo early stopping in uso), ecco dei "template" del codice da eseguire:

<p align="center">
    <strong>[Utility]</strong>
</p>

```python
$ python src/tracker.py --dataset={dataset} --model={model} --max_emission_step={max_emission_step} --trade_off={trade_off}
```
<br>
<p align="center">
    <strong>[EMA]</strong>
</p>

```python
$ python src/tracker.py --dataset={dataset} --model={model} --tolerance_step={tolerance_step} --smoothing_factor={smoothing_factor}
```
<br>

---

## 💻⚙️ Informazioni su hardware
Esperimenti effettutati sul seguente hardware:
* CPU: 1 x Intel Xeon Platinum 8358
* GPU: 1 x NVIDIA A100
* RAM: 512GB DDR4 (3200 MHz)
<br><br>

---

## 📊🤖 Informazioni su modelli e dataset
Gli esperimenti sono stati effettuati su i seguenti dataset e modelli di raccomandazione SOTA:

* Dataset: **amazon_books_60core** e **movielens_1m**
* Modelli: **BPR**, **CFKG**, **CKE**, **DMF**, **KGCN**, **KGNNLS**, **LINE**, **MultiDAE**, **LightGCN**, **NFCF** e **DGCF**
<br><br>

---

## 👨‍🎓 Informazioni autore
**Vincenzo Monopoli**<br>

Corso di Laurea in Informatica<br>
Università degli Studi di Bari “Aldo Moro”<br>

Tesi triennale in "Metodi per il Ritrovamento dell’Informazione"<br>
Anno Accademico 2023/2024<br>

📫 Contatti: [monovinci@gmail.com](mailto:monovinci@gmail.com)<br> 
🔗 GitHub: [github.com/Vincy02](https://github.com/Vincy02)