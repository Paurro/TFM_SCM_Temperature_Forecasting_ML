import os
import json
import subprocess

# 🔁 Diccionari: fitxer de config -> path de resultats
# (Aquests "save_dir" ja estan dins de cada fitxer .json)

# CONFIGS = {
#     "exps_finals.json",
#     "exps_serie_A.json",
#     "exps_serie_B.json",
#     "exps_serie_C.json",
#     "exps_baseline.json"
# }

CONFIG_DIR = "configs"
LAUNCHER_SCRIPT = "launcher_experiments_lstm.py"

# 🔍 Llista tots els fitxers JSON dins la carpeta
CONFIGS = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json") and not f.startswith("temp_")]

for config_name in CONFIGS:
    config_path = os.path.join(CONFIG_DIR, config_name)

    # 🔧 Llegeix el fitxer de config original
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # 🔧 Assegura que el directori de resultats existeix
    save_dir = config_data.get("save_dir", "resultats/default")
    os.makedirs(save_dir, exist_ok=True)

    # 📝 Crea una còpia temporal del fitxer (amb el save_dir ja inclòs)
    temp_config_path = os.path.join(CONFIG_DIR, f"temp_{config_name}")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n🚀 Executant experiments de: {config_name}")
    subprocess.run(["python", LAUNCHER_SCRIPT, temp_config_path])

    # 🧹 Esborrem el fitxer temporal (opcional)
    os.remove(temp_config_path)
