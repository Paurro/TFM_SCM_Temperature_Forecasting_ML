# TFM - Predicció de temperatura a curt termini amb *Machine Learning*

Aquest repositori conté el codi, les dades i la memòria del **Treball de Fi de Màster (TFM)** de **Pau Rodrigo Parés**, titulat:  
**"Predicció de temperatura a curt termini utilitzant machine learning"**

El projecte s’ha realitzat dins la línia de **Recerca Aplicada i Modelització (RAM)** del **Servei Meteorològic de Catalunya (SMC)** i el **Màster en Modelització per a la Ciència i l’Enginyeria** de la **Universitat Autònoma de Barcelona (UAB)**.

## 🧠 Objectiu del projecte

Aquest projecte analitzar la viabilitat de models de *machine learning* i, explora i compara diferents tècniques per a la predicció horària de la temperatura a curt termini:

- 🌧️ **Models probabilístics**: Cadenes de Markov per a precipitació
- 📊 **Models estadístics**: ARIMA i SARIMA
- 🧠 **Models de Deep Learning**: Xarxes LSTM (*Long Short-Term Memory*)

Les dades provenen de l’estació meteorològica de **la Bonaigua (XEMA)**, i cobreixen el període 1998–2024.

## 📁 Estructura del repositori

```
TFM_SCM_Temperature_Forecasting_ML/
│
├── environment.yml              # Entorn Conda amb totes les dependències
├── requirements.txt             # Alternativa amb pip
│
├── models_prediccio/
│   ├── 0_Data/                  # Dades originals i pre-processades
│   ├── 1_model_markov/          # Model de cadenes de Markov
│   ├── 2_model_arima/           # Models ARIMA i SARIMA
│   └── 3_model_lstm/            # Xarxes LSTM i experiments
│
├── Bibliografia/                # Articles científics consultats (PDFs)
├── TFM_PauRodrigo_*.pdf         # Memòria final del treball
└── README.md                    # Aquest fitxer
```

## 🛠️ Llibreries principals

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `statsmodels`, `tensorflow`, `keras`
- `joblib`, `plotly`, `tqdm`

## ⚙️ Com executar-ho

Per garantir la reproducció dels experiments:

```bash
conda env create -f environment.yml
conda activate tfm-forecast
```

## 🚀 Experiments i notebooks

Els notebooks Jupyter es troben a `models_prediccio/`, agrupats per model. Cada un inclou:

- Preprocessament i preparació de dades
- Entrenament i generació de prediccions
- Càlcul de mètriques (RMSE, MAE, etc.)
- Gràfics comparatius de resultats

Els scripts `.py` modularitzen la pipeline per facilitar l’execució i reutilització.

## 📄 Memòria del TFM

La memòria completa es troba al fitxer:

```
TFM_PauRodrigo_Temperature_Forecasting_ML_SCM.pdf
```

## 📚 Bibliografia

Els articles de referència utilitzats estan disponibles a la carpeta `Bibliografia/`.

## 📜 Crèdits

Treball desenvolupat per **Pau Rodrigo Parés**, amb la supervisió del **Servei Meteorològic de Catalunya (SMC)** durant el curs 2024–2025, dins del **Màster en Modelització per a la Ciència i l'Enginyeria** (UAB).
