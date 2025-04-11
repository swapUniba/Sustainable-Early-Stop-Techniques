import pandas as pd

dataset = ["amazon_books_60core", "movielens_1m"]
_index = 0
path = dataset[_index] + "/" + dataset[_index] + ".inter"
inter_df = pd.read_csv(path, sep="\t", header=None, dtype={2: str}, skiprows=1)
# prima riga contiene info nomi delle colonne (da quanto ho visto) => per questo skip

if len(inter_df.columns) > 2:
    interazioni_nulle = inter_df[2].isnull().sum()
    print(f"Interazioni nulle (valori mancanti): {interazioni_nulle}")
    interazioni_zero = (inter_df[2] == 0).sum()
    print(f"Interazioni nulle (valori zero): {interazioni_zero}")

valori_unici = inter_df[2].unique()
print("Valori unici:")
print(valori_unici)

print(inter_df.head(5))