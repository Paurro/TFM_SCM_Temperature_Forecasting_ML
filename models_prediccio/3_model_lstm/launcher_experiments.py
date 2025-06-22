import os
import json
import pandas as pd
from lstm_functions import executar_experiment, construir_nom_experiment
import sys



# -------------------------# 
# 📂 0. Importar mòduls 
# -------------------------
# Permet passar el nom del fitxer de configuració com a argument
config_path = sys.argv[1] if len(sys.argv) > 1 else "params_experiment.json"
with open(config_path) as f:
    config = json.load(f)



# -------------------------
# 📂 1. Carregar dades
# -------------------------
__path__ = os.getcwd()
carpeta_dades = '../0_Data/Dades_T_estacions_xema_Z1'
fitxer_dades = 'SCM_T_Z1_2020_2024.csv'
path_dades = os.path.join(__path__, carpeta_dades, fitxer_dades)

df_lstm = pd.read_csv(path_dades)
df_lstm['data'] = pd.to_datetime(df_lstm['data'], utc=True)


# -------------------------
# ⚙️ 2. Carregar configuració
# -------------------------
with open("params_experiment.json") as f:
    config = json.load(f)

save_dir = config.get("save_dir", "resultats")
assignacio_num = config.get("assignacio_num", True)


# -------------------------
# 🧪 3. Construir experiments
# -------------------------
if "experiments" in config:
    experiments_list = config["experiments"]
else:
    # Generar experiments a partir de llistes paral·leles
    n_experiments = len(config["WINDOW_SIZE"])  # Es pressuposa mateixa mida a totes

    experiments_list = []
    for i in range(n_experiments):
        exp = {
            "window_size": config["WINDOW_SIZE"][i],
            "n_outputs": config["N_OUTPUTS"][i],
            "n_layers": config["N_LAYERS"][i],
            "n_units": config["N_UNITS"][i],
            "lookahead": config.get("LOOKAHEAD", 0),
            "dropout_rate": config.get("DROPOUT_RATE", 0.2),
            "batch_size": config.get("BATCH_SIZE", 128),
            "epochs": config.get("EPOCHS", 50),
            "patience": config.get("PATIENCE", 5),
            "seed": config.get("SEED", 42),
            "predictions_mod": config.get("PREDICTIONS_MOD", ["pred_batch"])
        }
        experiments_list.append(exp)


# -------------------------
# 🚀 4. Lançar experiments
# -------------------------
for i, params in enumerate(experiments_list):
    if assignacio_num:
        nom_experiment = f"experiment_{i}"
    else:
        nom_experiment = construir_nom_experiment(params)

    save_path = os.path.join(save_dir, nom_experiment)

    if os.path.exists(os.path.join(save_path, "model.h5")):
        print(f"❌ Experiment ja existeix: {nom_experiment}")
        continue

    try:
        print(f"🚀 Executant experiment {i + 1}/{len(experiments_list)}: {nom_experiment}")
        executar_experiment(
            df=df_lstm,
            params=params,
            save_path=save_path,
            mostrar_val=False,
            col_preds=params.get("predictions_mod", ["pred_batch"])
        )
        print(f"✅ Experiment {nom_experiment} completat\n")

    except Exception as e:
        print(f"⚠️  ERROR a l'experiment {nom_experiment}: {e}")
