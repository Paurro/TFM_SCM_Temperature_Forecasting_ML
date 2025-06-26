import os
import json
import pandas as pd
import sys

from arima_functions import executar_experiment_arima, construir_nom_experiment_arima


# import warnings
# from statsmodels.tools.sm_exceptions import ConvergenceWarning

# # Silenciar warnings concrets
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=ConvergenceWarning)



# -------------------------#
# 📂 0. Llegir fitxer de configuració
# -------------------------
config_path = sys.argv[1] if len(sys.argv) > 1 else "params_experiment_arima.json"
with open(config_path) as f:
    config = json.load(f)

# -------------------------#
# 📂 1. Carregar dades
# -------------------------
__path__ = os.getcwd()
carpeta_dades = '../0_Data/Dades_T_estacions_xema_Z1'
fitxer_dades = 'SCM_T_Z1_2020_2024.csv'
path_dades = os.path.join(__path__, carpeta_dades, fitxer_dades)

df = pd.read_csv(path_dades)
df['data'] = pd.to_datetime(df['data'], utc=True)

# -------------------------#
# ⚙️ 2. Configuració general
# -------------------------
save_dir = config.get("save_dir", "resultats_arima")
assignacio_num = config.get("assignacio_num", False)
rolling = config.get("rolling", False)

# -------------------------#
# 🧪 3. Construir experiments
# -------------------------
# if "experiments" in config:
#     experiments_list = config["experiments"]
# else:
#     n_experiments = len(config["p"])  # mateixa mida per p, d, q

#     experiments_list = []
#     for i in range(n_experiments):
#         exp = {
#             "id": config.get("id", [None]*n_experiments)[i],
#             "data_inici_pred": config["data_inici_pred"][i],
#             "data_final_pred": config.get("data_final_pred", [None]*n_experiments)[i],
#             "dies_entrenament": config["dies_entrenament"][i],
#             "p": config["p"][i],
#             "d": config["d"][i],
#             "q": config["q"][i],
#             "P": config.get("P", [None]*n_experiments)[i],
#             "D": config.get("D", [None]*n_experiments)[i],
#             "Q": config.get("Q", [None]*n_experiments)[i],
#             "s": config.get("s", [None]*n_experiments)[i],
#             "rolling": rolling,
#             "n_passos": config.get("n_passos", [None]*n_experiments)[i],
#             "pas_pred": config.get("pas_pred", [1]*n_experiments)[i],
#             "plot_dies_ant": config.get("plot_dies_ant", 0)
#         }
#         experiments_list.append(exp)

experiments_list = config["experiments"]


# -------------------------#
# 🚀 4. Lançar experiments
# -------------------------
for i, params in enumerate(experiments_list):
    # ✨ INCORPORACIÓ: ús del prefix ID si està present
    if assignacio_num:
        nom_experiment = f"experiment_{i}"
    else:
        prefix = params.get("id", f"exp{i:02d}")
        nom_experiment = construir_nom_experiment_arima(params, prefix=prefix)

    save_path = os.path.join(save_dir, nom_experiment)

    if os.path.exists(os.path.join(save_path, "metrics.csv")):
        print(f"❌ Experiment ja existeix: {nom_experiment}")
        continue

    try:
        print(f"🚀 Executant experiment {i + 1}/{len(experiments_list)}: {nom_experiment}")
        executar_experiment_arima(df=df, params=params, save_root=save_dir)
        print(f"✅ Experiment {nom_experiment} completat\n")

    except Exception as e:
        print(f"⚠️  ERROR a l'experiment {nom_experiment}: {e}")

