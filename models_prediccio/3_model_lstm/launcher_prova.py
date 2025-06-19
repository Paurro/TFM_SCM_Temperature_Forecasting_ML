import os
import pandas as pd
import itertools
from lstm_functions import executar_experiment, construir_nom_experiment

# === PREPARACIÓ DE DADES ===

__path__ = os.getcwd()
carpeta_dades = '../0_Data/Dades_T_estacions_xema_Z1'
fitxer_dades_csv_2020_2024 = 'SCM_T_Z1_2020_2024.csv'
path_dades = os.path.join(__path__, carpeta_dades, fitxer_dades_csv_2020_2024)

df_lstm = pd.read_csv(path_dades)
df_lstm['data'] = pd.to_datetime(df_lstm['data'], utc=True)


# === DEFINICIÓ COMBINACIONS PETITES PER PROVA ===

WINDOW_SIZES = [24]
N_OUTPUTS = [1, 6]
N_LAYERS = [1]
N_UNITS = [16]     # 👈 Poc pes per anar ràpid
DROPOUTS = [0.0]
BATCH_SIZE = 32
EPOCHS = 3         # 👈 Només 3 èpoques per anar ràpid

combinacions = list(itertools.product(WINDOW_SIZES, N_OUTPUTS, N_LAYERS, N_UNITS, DROPOUTS))


# === EXECUCIÓ DELS EXPERIMENTS ===

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

    save_path = None  # Es genera automàticament

    try:
        print(f"\n🚀 [{i+1}/{len(combinacions)}] Entrenant: win={win}, out={out}, lay={lay}, uni={uni}, drop={drop}")
        executar_experiment(df=df_lstm, params=params, save_path=save_path, mostrar_val=False,col_preds=['pred_iter'])
        print(f"✅ Experiment completat.\n")
    except Exception as e:
        print(f"⚠️  Error durant l'experiment {i+1}: {e}")
