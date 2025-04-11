import numpy as np
import pandas as pd
num_epochs = 200

np.random.seed(42)

# Generazione accuracy con crescita logaritmica
epochs = np.arange(1, num_epochs + 1)
mmr_values = np.clip(0.2 + 0.5 * np.log1p(epochs) / np.log1p(num_epochs), 0, 1)  # logaritmica normalizzata
mmr_values += np.random.normal(0, 0.01, num_epochs)  # rumore

# Generazione CO2
base_co2_per_epoch = 50  # g di CO2 in fase iniziale
co2_decay_factor = 0.995  # decrescita per miglioramenti di efficienza
co2_values_g = base_co2_per_epoch * (co2_decay_factor ** epochs) + np.random.normal(0, 10, num_epochs)  # rumore

# gr ==> Kg
co2_values_kg = co2_values_g / 100000

df_realistic = pd.DataFrame({
    'Epoch': epochs,
    'MMR@10': np.clip(mmr_values, 0, 1),
    'CO2 Emissions (kg)': np.maximum(co2_values_kg, 0)
})

df_realistic['Cumulative CO2 Emissions (Kg)'] = df_realistic['CO2 Emissions (kg)'].cumsum()

#print(df_realistic.head())

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()

# Salvo tutto
mmr_values_list = df_realistic['MMR@10'].tolist()
co2_values_list = df_realistic['Cumulative CO2 Emissions (Kg)'].tolist()

output_file = 'emission_data.txt'

with open(output_file, 'w') as f:
    f.write(f"acc = {mmr_values_list}\n")
    f.write(f"co2 = {co2_values_list}\n")

# Plot
# Acc
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Acc/Metrica', color='tab:red')
ax1.plot(df_realistic['MMR@10'], color='tab:red', label='Acc')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Emission
ax2 = ax1.twinx()
ax2.set_ylabel('Emission (Kg CO2)', color='tab:blue')
ax2.plot(df_realistic['Cumulative CO2 Emissions (Kg)'], color='tab:blue', label='Power')
ax2.tick_params(axis='y', labelcolor='tab:blue')

fig.tight_layout(pad=2.0)
plt.title('acc - Co2')
plt.show()