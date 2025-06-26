# TFM_SCM_Temperature_Forecasting_ML
TFM de predicció de temperatura a curt termini utilitzant models de Machine Learning

# Predicció de temperatura a curt termini amb models seqüencials
Aquest repositori recull el codi font, dades i figures associades al **Treball de Fi de Màster (TFM)** de Pau Rodrigo:  
**"Predicció de temperatura a curt termini amb xarxes neuronals LSTM"**  
realitzat en col·laboració amb el Servei Meteorològic de Catalunya (SMC) dins la línia de Recerca Aplicada i Modelització (RAM).

## 🧠 Objectiu del projecte

Explorar i comparar diferents mètodes per a la predicció horària de temperatura a curt termini mitjançant:

- Xarxes neuronals LSTM (Long Short-Term Memory)
- Models estadístics clàssics ARIMA i SARIMA
- Cadenes de Markov per predicció qualitativa de precipitació

Els models han estat entrenats amb dades reals de l’estació meteorològica de la Bonaigua (XEMA) entre 1998 i 2024.

## 📁 Estructura del repositori

```
TFM_SCM_Temperature_Forecasting_ML/
├── Bibliografia/           # Articles científics consultats (PDFs)
├── Experiments/            # Notebooks d’experiments per model
├── Figures/                # Gràfics i figures generades pel TFM
├── Scripts/                # Funcions Python utilitzades als notebooks
├── Data/                   # Dades d’entrada (temperatura, precipitació…)
├── requirements.txt        # Llista de paquets per a reproducció
├── environment.yml         # Fitxer d’entorn per Conda
└── README.md               # Aquest fitxer
```

## 🛠️ Llibreries principals

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `statsmodels`, `tensorflow`, `keras`
- `joblib`, `plotly`, `tqdm`

## 📦 Entorn recomanat

Per garantir la reproducció dels experiments, pots crear un entorn Conda:

```bash
conda env create -f environment.yml
conda activate TFM_venv
```

## 🚀 Execució dels experiments

Els notebooks es troben dins la carpeta `Experiments/` i estan agrupats per model. Cada un inclou:

- Preprocessament i preparació de dades
- Entrenament i predicció
- Avaluació amb mètriques (RMSE, MAE, etc.)
- Visualització dels resultats

## 📜 Crèdits

Aquest treball ha estat desenvolupat per **Pau Rodrigo** durant el Màster en Modelització per a la Ciència i l'Enginyeria (UAB), amb suport del **Servei Meteorològic de Catalunya (SMC)**.
