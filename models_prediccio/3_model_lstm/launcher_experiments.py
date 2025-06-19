
# launcher.py

import itertools
import os
import pandas as pd
from lstm_functions import executar_experiment, construir_nom_experiment


# Carrega les dades
# Preparació de dades
__path__ = os.getcwd()
carpeta_dades = '../0_Data/Dades_T_estacions_xema_Z1'
fitxer_dades_csv_2020_2024 = 'SCM_T_Z1_2020_2024.csv'
path_dades = os.path.join(__path__, carpeta_dades, fitxer_dades_csv_2020_2024)

df_lstm = pd.read_csv(path_dades)
df_lstm['data'] = pd.to_datetime(df_lstm['data'], utc=True)



# Defineix valors a provar
WINDOW_SIZES = [24, 48]
N_OUTPUTS = [1, 6]
N_LAYERS = [1, 2, 3]
N_UNITS = [32, 64]
DROPOUTS = [0.0, 0.1]
BATCH_SIZE = 32
EPOCHS = 30

# Genera totes les combinacions
combinacions = list(itertools.product(WINDOW_SIZES, N_OUTPUTS, N_LAYERS, N_UNITS, DROPOUTS))

for i, (win, out, lay, uni, drop) in enumerate(combinacions):

    params = {
        'window_size': win,
        'n_outputs': out,
        'n_layers': lay,
        'n_units': uni,
        'dropout_rate': drop,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS
    }

    # Ruta de guardat automàtica
    nom = construir_nom_experiment(params)
    save_path = os.path.join("resultats", nom)

    # Si ja existeix, el saltem
    if os.path.exists(os.path.join(save_path, "model.h5")):
        print(f"❌ Experiment {nom} ja existeix, es salta.")
        continue

    try:
        print(f"🚀 Executant experiment {i+1}/{len(combinacions)}: {nom}")
        executar_experiment(df=df, params=params, save_path=save_path, mostrar_val=False)
        print(f"✅ Experiment {nom} completat\n")
    except Exception as e:
        print(f"⚠️  ERROR a l'experiment {nom}: {e}")
