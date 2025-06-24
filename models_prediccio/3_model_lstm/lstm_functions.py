# ===============================================================
# Funcions per a la predicció de temperatura amb xarxes LSTM
# ===============================================================
# Autor: Pau Rodrigo
# Projecte: TFM - Predicció de temperatura a curt termini amb LSTM
# Data: Juny 2025
#
# Descripció:
# Aquest fitxer conté funcions modulars per a la preparació de dades,
# entrenament de models LSTM, prediccions (batch i iteratives) i càlcul
# de mètriques. Està pensat per a ser importat des d'un notebook principal.
#
# Llibreries requerides:
# - numpy
# - pandas
# - tensorflow / keras
# - scikit-learn
# - matplotlib (només si es fan gràfics)
#
#
# Ús recomanat:
# from lstm_functions import*
# ===============================================================

# ============================
# Llibreries estàndard
# ============================
import os
import random

# ============================
# Llibreries científiques
# ============================
import numpy as np
import pandas as pd
import json

# ============================
# Visualització
# ============================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ============================
# Machine Learning
# ============================
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================
# TensorFlow / Keras
# ============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================
# Altres (només si uses display() dins funcions)
# ============================
from IPython.display import display




# ================================================================
# Funcions per la preparació de dades 
# ================================================================

# Funció per escalar les dades a milers (graficació)

def escala_mil(x, pos):
    """
    Funció per formatar els ticks de l’eix Y multiplicant per 1000.
    És útil quan vols mostrar valors petits (com MSE en [0, 0.01]) en una escala més llegible.

    Args:
        x (float): valor del tick original (per exemple, 0.0013).
        pos (int): posició del tick en l’eix (0, 1, 2...). No s’utilitza aquí, però és necessari
                   perquè FuncFormatter sempre crida la funció amb dos arguments.

    Returns:
        str: valor formatat com a string, multiplicat per 1000 i amb 1 decimal (ex: '1.3').
    """
    val = x * 1000
    return f'{val:.1f}'


# Funció per separar les dades en train, val i test

def split_dades(df_lstm):
    """
    Separa un DataFrame de dades temporals en conjunts de train, validació i test,
    basant-se en dates límit relatives al màxim de la columna 'data'.

    Args:
        df_lstm (pd.DataFrame): DataFrame amb una columna 'data' i una columna 'valor'.
        limit_train (int): mesos per definir el límit del conjunt de train.
        limit_val (int): mesos per definir el límit del conjunt de validació.

    Returns:
        df_train, df_val, df_test: DataFrames separats per train, validació i test.
    """
    
    # Comprovem que el DataFrame tingui les columnes necessàries
    if 'data' not in df_lstm.columns or 'valor' not in df_lstm.columns:
        raise ValueError("El DataFrame ha de tenir les columnes 'data' i 'valor'.")


    # Definim les dates límit per la separació
    data_max = df_lstm['data'].max()

    data_limit_train = data_max - pd.DateOffset(months= 6)  # Límit inicial del train
    data_limit_val = data_max - pd.DateOffset(months= 3)    # Límit inicial de la validació

    # Separem els datasets
    df_train = df_lstm[df_lstm['data'] <= data_limit_train].copy().reset_index(drop=True)
    df_val = df_lstm[(df_lstm['data'] > data_limit_train) & (df_lstm['data'] <= data_limit_val)].copy().reset_index(drop=True)
    df_test = df_lstm[df_lstm['data'] > data_limit_val].copy().reset_index(drop=True)

    return df_train, df_val, df_test



# Crea una funció per escalar les dades

def escalar_dades(df_train, df_val, df_test, columna='valor', verbose=True):
    """
    Escala els valors d'una columna numèrica utilitzant MinMaxScaler.
    L'ajust es fa només sobre el conjunt de train, i després s'aplica als altres.

    Parameters:
    - df_train, df_val, df_test: DataFrames amb la columna a escalar
    - columna: nom de la columna a escalar (per defecte 'valor')

    Returns:
    - df_train, df_val, df_test: DataFrames amb una nova columna 'columna_scaled'
    - scaler: objecte MinMaxScaler ja entrenat
    """
    
    # Importem el Scaler
    scaler = MinMaxScaler()

    # Escalar només sobre train i transformar val i test
    df_train[f'{columna}_scaled'] = scaler.fit_transform(df_train[[columna]])
    df_val[f'{columna}_scaled'] = scaler.transform(df_val[[columna]])
    df_test[f'{columna}_scaled'] = scaler.transform(df_test[[columna]])


    # Observem com queden les dades
    print('✅ Escalat completat:')
    print("\n")

    if verbose:

        print('Train dataset shape:', df_train.shape)
        # display(df_train.head())

        print('Validation dataset shape:', df_val.shape)
        # display(df_val.head())

        print('Test dataset shape:', df_test.shape)
        # display(df_test.head())

    return df_train, df_val, df_test, scaler




# Creem una funció per crear sequences per LSTM d'entrada

def create_sequences(series, window_size, n_outputs=1, n_slide=None, lookahead=0):
    """
    Crea seqüències d'entrada i sortida per predicció simple o multi-output, 
    amb suport per desplaçament entre finestres (n_slide) i predicció desfasada (lookahead).

    Args:
        series (array): sèrie temporal escalada.
        window_size (int): llargada de la finestra d'entrada.
        n_outputs (int): nombre de passos a predir (per defecte 1).
        n_slide (int): quant avancem la finestra a cada iteració (per defecte 1).
        lookahead (int): passos entre el final de la finestra i la primera predicció (per defecte 0).

    LSTM espera una entrada en 3 dimensions:
        (n_samples, window_size, n_features)
        On:
            - n_samples = nombre de finestres que hem generat
            - window_size = longitud de cada finestra (número de valors consecutius)
            - n_features = nombre de variables per timestep (en aquest cas, 1 sola: la temperatura)

    Returns:
        X (np.array): seqüències d'entrada, forma (samples, window_size, 1).
        y (np.array): seqüències de sortida, forma (samples, n_outputs) si n_outputs > 1, 
                      o (samples, 1) si n_outputs = 1.
    """

    if n_slide is None:
        n_slide = n_outputs

    X, y = [], []

    max_i = len(series) - lookahead - n_outputs
    i = window_size

    while i <= max_i:
        seq_x = series[i - window_size:i]
        seq_y = series[i + lookahead : i + lookahead + n_outputs]

        X.append(seq_x)
        if n_outputs == 1:
            y.append(seq_y[0])
        else:
            y.append(seq_y)

        i += n_slide

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    if n_outputs == 1:
        y = y.reshape(-1, 1)

    return X, y




# ================================================================
# Funcions per a la creació i entrenament del model LSTM
# ================================================================

def definir_model_lstm(
    window_size,       # mida de la finestra temporal (timesteps)
    n_features,        # nombre de variables d'entrada (features)
    n_outputs,         # nombre de valors a predir
    n_layers=3,        # nombre total de capes LSTM
    n_units=64,        # neurones per cada capa LSTM
    dropout_rate=0.2,  # percentatge de neurones a desactivar (Dropout)
    optimizer='adam',  # optimitzador per compilar el model
    loss='mse'         # funció de pèrdua (ideal 'mse' per regressió)
):
    
    """
    Crea i compila un model LSTM seqüencial personalitzable.

    Args:
        window_size (int): mida de la finestra temporal (timesteps).
        n_features (int): nombre de variables (features) d'entrada.
        n_outputs (int): nombre de valors a predir.
        n_layers (int): nombre total de capes LSTM.
        n_units (int): neurones per capa LSTM.
        dropout_rate (float): percentatge de neurones a desactivar si s'utilitza Dropout.
        optimizer (str): optimitzador per compilar el model.
        loss (str): funció de pèrdua per compilar el model.

    Returns:
        model (keras.Sequential): model LSTM ja compilat.
    """
    model = Sequential()

    # Capa inicial LSTM (amb input_shape)
    # Si hi ha més d'una capa, cal que retorni seqüència
    model.add(LSTM(n_units, return_sequences=(n_layers > 1), input_shape=(window_size, n_features)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Capes intermèdies (si n_layers >= 3)
    for _ in range(n_layers - 2):
        model.add(LSTM(n_units, return_sequences=True))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Última capa LSTM (si n_layers >= 2)(sense return_sequences)
    if n_layers > 1:
        model.add(LSTM(n_units))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Capa de sortida (1 neurona per cada output)
    model.add(Dense(n_outputs))

    # Compilació
    model.compile(
        optimizer=optimizer,
        loss=loss
        )

    return model



def train_model(
    model,              # model LSTM compilat
    X_train, y_train,   # conjunts d'entrenament
    X_val, y_val,       # conjunts de validació
    epochs=50,          # nombre d'iteracions d'entrenament (Backpropagation)
    batch_size=32,      # mida del lot, nombre de mostres processades abans de l'actualització dels pesos.
    patience=5,         # paciència per l'EarlyStopping. Nombre de epoques sense millora abans d'aturar l'entrenament.
    shuffle=False,      # si es vol barrejar les dades (normalment False en seqüències)
    seed=42,            # seed per assegurar la reproduïbilitat
    summary=True       # si es vol mostrar el resum del model al final
):
    """
    Entrena un model LSTM amb validació i EarlyStopping, mantenint l'ordre si es vol.

    Args:
        model: model LSTM compilat.
        X_train, y_train: conjunts d'entrenament.
        X_val, y_val: conjunts de validació.
        epochs (int): nombre d'èpoques.
        batch_size (int): mida del lot.
        patience (int): paciència per l'EarlyStopping.
        shuffle (bool): si es vol barrejar les dades (normalment False en seqüències).
        seed (int): seed per assegurar la reproduïbilitat.

    Returns:
        history: historial de l'entrenament.
    """

    # Fixem llavors per a la reproduïbilitat
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


    # EarlyStopping per evitar overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    print('Entrenant el model LSTM')

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        shuffle=shuffle,
        verbose=1
    )

    if summary:
        print("\nModel Summary:")
        model.summary()

    return history




def plot_loss_train_val(history,show=True):

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(history.history['loss'], label='Pèrdua Entrenament')
    ax.plot(history.history['val_loss'], label='Pèrdua Validació')
    ax.set_title("Evolució de la pèrdua durant l'entrenament")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (×10⁻³)")

    # Per ticks x cada 2 epochs
    ax.set_xticks(np.arange(0, len(history.history['loss']), 1))

    # Format ticks y amb 1 decimal
    ax.yaxis.set_major_formatter(FuncFormatter(escala_mil))

    # Força ticks a enters (però com fem decimals, pot ser opcional)
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))  # nbins controla nombre màxim ticks

    ax.legend()
    ax.grid(True)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ================================================================
# Funcions de predicció LSTM 1 output
# ================================================================

def prediccio_batch(model, X_test, df_test_pred, scaler, nom_columna='pred_batch',lookahead=0):
    """
    Fa una predicció batch (totes les finestres alhora), desescala les prediccions i les afegeix directament a df_test_pred.

    Args:
        model: model LSTM entrenat.
        X_test (np.array): finestres d’entrada per a la predicció (forma: (n_samples, window_size, 1)).
        df_test_pred (pd.DataFrame): DataFrame amb la columna 'valor' desescalada. Es modifica in-place.
        scaler: MinMaxScaler ajustat sobre les dades de train.
        nom_columna (str): nom de la columna on s’enganxaran les prediccions (per defecte 'pred_batch').

    Returns:
        df_test_pred (DataFrame): DataFrame amb la nova columna de predicció.
        y_pred_rescaled (np.array): prediccions desescalades (forma: (n_samples,)).
    """
    # Predicció i desescalat
    y_pred = model.predict(X_test, verbose=0)
    y_pred_rescaled = scaler.inverse_transform(y_pred).flatten()

    # Assignar al DataFrame (ignorant les primeres files sense prou context)
    window_size = X_test.shape[1]
    idx_valid = df_test_pred.index[window_size + lookahead:]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred



def prediccio_step_iterativa(model, X_test, df_test_pred, scaler, nom_columna='pred_iter', lookahead=0):
    """
    Fa una predicció multi-step iterativa, reinjectant cada predicció com a nou input,
    i afegeix les prediccions desescalades directament a df_test_pred.

    Args:
        model: model LSTM entrenat.
        X_test (np.array): finestres d’entrada (forma: (n_samples, window_size, 1)).
        df_test_pred (pd.DataFrame): DataFrame amb la columna 'valor' desescalada. Es modifica in-place.
        scaler: MinMaxScaler ajustat sobre les dades de train.
        nom_columna (str): nom de la columna on s’enganxaran les prediccions (per defecte 'pred_iter').

    Returns:
        df_test_pred (DataFrame): amb la nova columna de predicció iterativa afegida.
        y_pred_rescaled (np.array): prediccions desescalades (forma: (n_samples,)).
    """
    window_size = X_test.shape[1]
    n_passos = X_test.shape[0]

    seq = X_test[0].copy()  # Seqüència inicial escalada (window_size, 1)
    preds_scaled = []

    for _ in range(n_passos):
        input_seq = seq.reshape((1, window_size, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        preds_scaled.append(pred_scaled)
        seq = np.append(seq[1:], [[pred_scaled]], axis=0)

    # Desescalar les prediccions
    y_pred_rescaled = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Assignar les prediccions desescalades al final del DataFrame
    idx_valid = df_test_pred.index[window_size + lookahead : window_size + lookahead + n_passos]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred



def prediccio_iterativa_reinjection(model, X_test, df_test_pred, scaler, reinjeccio=5, nom_columna='pred_reinject', lookahead=0):
    """
    Fa una predicció iterativa amb reinjecció de valors reals cada 'reinjeccio' passos,
    i afegeix les prediccions desescalades al df_test_pred.

    Args:
        model: model LSTM entrenat.
        X_test (np.array): finestres d’entrada (forma: (n_samples, window_size, 1)).
        df_test_pred (pd.DataFrame): DataFrame amb la columna 'valor_scaled'. Es modifica in-place.
        scaler: MinMaxScaler ajustat sobre les dades de train.
        reinjeccio (int): nombre de passos entre reinjeccions de dades reals.
        nom_columna (str): nom de la columna on s’enganxaran les prediccions.

    Returns:
        df_test_pred (DataFrame): amb la nova columna de prediccions afegida.
        y_pred_rescaled (np.array): prediccions desescalades (forma: (n_samples,)).
    """
    window_size = X_test.shape[1]
    n_passos = X_test.shape[0]

    valors_scaled = df_test_pred['valor_scaled'].values
    preds_scaled = []

    # Inicialitzem amb valors reals escalats
    # seq = X_test[0].copy()  # Seqüència inicial (forma: (window_size, 1))
    seq = valors_scaled[:window_size].reshape(-1, 1).copy()
    
    for i in range(n_passos):
        input_seq = seq.reshape((1, window_size, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        preds_scaled.append(pred_scaled)

        if (i + 1) % reinjeccio == 0:
            start_real = i + 1
            end_real = start_real + window_size
            if end_real <= len(valors_scaled):
                seq = valors_scaled[start_real:end_real].reshape(-1, 1).copy()
            else:
                seq = np.append(seq[1:], [[pred_scaled]], axis=0)
        else:
            seq = np.append(seq[1:], [[pred_scaled]], axis=0)


    # Desescalar prediccions
    y_pred_rescaled = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Assignar al DataFrame
    n_preds = len(preds_scaled)
    idx_valid = df_test_pred.index[window_size + lookahead : window_size + lookahead + n_preds]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred




# ================================================================
# Funcions de predicció LSTM multi-output
# ================================================================


# Funció per fer prediccions multi-step

def prediccio_batch_multi(model, X_test, df_test, scaler, window_size, n_outputs, nom_columna='pred_batch',lookahead=0):

    """
    Fa prediccions multi-output de manera contínua i enganxa totes les prediccions al DataFrame original.

    Supòsits:
    - S'utilitzen seqüències creades amb `n_slide = n_outputs`, per tant NO hi ha solapament entre finestres.
    - Cada finestra prediu exactament els següents `n_outputs` valors, i la següent finestra continua on acaba l’anterior.

    Args:
        model: Model LSTM multi-output entrenat.
        X_test (np.array): Matriu d’entrada per a test (n_samples, window_size, 1).
        df_test (pd.DataFrame): DataFrame original amb les dades reals, conté almenys la columna 'valor'.
        scaler: MinMaxScaler utilitzat per escalar i desescalar les dades.
        window_size (int): Mida de la finestra d’entrada per a cada seqüència.
        n_outputs (int): Nombre de passos que prediu el model (outputs per finestra).
        nom_columna (str): Nom de la columna on es guardaran les prediccions desescalades.
        lookahead (int): Passos entre el final de la finestra i la primera predicció (per defecte 0).

    Retorna:
        df_test amb la nova columna `nom_columna` que conté les prediccions (amb NaNs on no es pot predir).
    """
    # 1. Fer la predicció batch
    y_pred = model.predict(X_test, verbose=0)

    # 2. Desescalar les prediccions (per tornar a °C)

    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_pred_rescaled = y_pred_rescaled.flatten()  # Aplanar per tenir un vector de prediccions
    
    # 3. Inicialitzem la nova columna amb NaNs
    df_test[nom_columna] = np.nan

    # Calcul rang disponible per a les prediccions
    idx_inici = window_size + lookahead # Inici de les prediccions després de la finestra inicial
    dispo = len(df_test) - idx_inici  # Espai disponible per a les prediccions
    usable_preds = min(len(y_pred_rescaled), dispo)  # Nombre de prediccions que podem utilitzar

    # 5. Assignació segura dels valors
    df_test.iloc[idx_inici:idx_inici + usable_preds, df_test.columns.get_loc(nom_columna)] = y_pred_rescaled[:usable_preds]

    # 6. Avís si hi ha truncament
    if usable_preds < len(y_pred_rescaled):
        print(f"⚠️ {len(y_pred_rescaled) - usable_preds} valors de predicció no s'han col·locat per falta d'espai. Prediccions truncades a {usable_preds} valors, per que superaven l'espai disponible a df_test.")

    return df_test



def prediccio_step_iterativa_multi(model, X_test, df_test_pred, scaler, nom_columna='pred_iter', lookahead=0):
    """
    Fa una predicció multi-step iterativa (multi-output), reinjectant les prediccions com a nova entrada,
    i afegeix les prediccions desescalades directament a df_test_pred.

    Assumim que el model prediu diversos passos (multi-output), i que X_test conté una sola seqüència inicial.

    Args:
        model: model LSTM entrenat.
        X_test (np.array): finestres d’entrada (forma: (n_samples, window_size, 1)).
        df_test_pred (pd.DataFrame): DataFrame amb la columna 'valor' desescalada. Es modifica in-place.
        scaler: MinMaxScaler ajustat sobre les dades de train.
        nom_columna (str): nom de la columna on s’enganxaran les prediccions (per defecte 'pred_iter').
        lookahead (int): passos entre el final de la finestra i la primera predicció (per defecte 0).

    Returns:
        df_test_pred (DataFrame): amb la nova columna de predicció iterativa afegida.
    """
    window_size = X_test.shape[1]
    n_outputs = model.output_shape[-1]

    seq = X_test[0].copy()  # Seqüència inicial escalada (window_size, 1)
    preds_scaled = []

    n_preds_total = len(df_test_pred) - window_size
    n_steps = n_preds_total // n_outputs

    for _ in range(n_steps):
        input_seq = seq.reshape((1, window_size, 1))  # Afegim dimensió batch
        pred_scaled = model.predict(input_seq, verbose=0)[0]  # (n_outputs,)
        preds_scaled.extend(pred_scaled)

        # Afegim les prediccions escalades al final de la seqüència
        pred_scaled_reshaped = pred_scaled.reshape(-1, 1)
        seq = np.concatenate([seq[n_outputs:], pred_scaled_reshaped], axis=0)

    # Desescalar totes les prediccions
    y_pred_rescaled = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Assignar les prediccions al DataFrame
    idx_valid = df_test_pred.index[window_size + lookahead : window_size + lookahead + len(y_pred_rescaled)]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred



def prediccio_iterativa_reinjection_multi(model, X_test, df_test_pred, scaler, reinjeccio=5, nom_columna='pred_reinject',lookahead=0):
    """
    Fa una predicció iterativa multi-output amb reinjecció de valors reals cada 'reinjeccio' passos.
    Afegeix les prediccions desescalades al df_test_pred.

    Args:
        model: model LSTM multi-output entrenat.
        X_test (np.array): finestres d’entrada (forma: (n_samples, window_size, 1)).
        df_test_pred (pd.DataFrame): DataFrame amb la columna 'valor_scaled'. Es modifica in-place.
        scaler: MinMaxScaler ajustat sobre les dades de train.
        reinjeccio (int): cada quants passos es reinjecta el valor real.
        nom_columna (str): nom de la columna on s’enganxaran les prediccions.

    Returns:
        df_test_pred (DataFrame): amb la nova columna de prediccions afegida.
    """
    window_size = X_test.shape[1]
    n_outputs = model.output_shape[-1]
    valors_scaled = df_test_pred['valor_scaled'].values
    preds_scaled = []

    # Inicialitzem amb la primera finestra real
    seq = valors_scaled[:window_size].reshape(-1, 1).copy()

    # Nombre total de passos de predicció (amb salt de n_outputs)
    n_preds_total = len(df_test_pred) - window_size
    n_steps = n_preds_total // n_outputs

    for i in range(n_steps):
        input_seq = seq.reshape((1, window_size, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0]  # (n_outputs,)
        preds_scaled.extend(pred_scaled)

        # Reinjecció cada 'reinjeccio' passos
        if (i + 1) % reinjeccio == 0:
            start_real = i * n_outputs
            end_real = start_real + window_size
            if end_real <= len(valors_scaled):
                seq = valors_scaled[start_real:end_real].reshape(-1, 1).copy()
            else:
                pred_reshaped = pred_scaled.reshape(-1, 1)
                seq = np.concatenate([seq[n_outputs:], pred_reshaped], axis=0)
        else:
            pred_reshaped = pred_scaled.reshape(-1, 1)
            seq = np.concatenate([seq[n_outputs:], pred_reshaped], axis=0)

    # Desescalar i inserir al DataFrame
    y_pred_rescaled = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    idx_valid = df_test_pred.index[window_size + lookahead : window_size + lookahead + len(y_pred_rescaled)]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred



# ================================================================
# Funcions de càlcul de mètriques i gràfics
# ================================================================


def calcular_metriques(df_test_pred, col_real='valor', col_preds=['pred_batch', 'pred_iter', 'pred_reinject']):
    """
    Calcula RMSE, MSE i MAE per diferents columnes de predicció respecte a una columna real.

    Args:
        df_test_pred (pd.DataFrame): DataFrame amb les columnes de valors reals i prediccions.
        col_real (str): Nom de la columna amb els valors reals.
        col_preds (list): Llista amb noms de les columnes de predicció.
        window_size (int): Mida de la finestra per alinear les dades.

    Returns:
        pd.DataFrame: Taula amb RMSE, MSE i MAE per cada mètode de predicció.
    """
    metriques = {'Mètrica': ['RMSE', 'MSE', 'MAE']}

    # Iterem i calculem les mètriques per cada columna de predicció
    for col in col_preds:

        # Comprovem que la columna de predicció existeixi
        if col not in df_test_pred.columns:
            print(f"⚠️ Avís: la columna '{col}' no existeix a df_test_pred. Es descarta.")
            continue

        # Eliminem files amb NaNs (pot ser degut a window_size, lookahead, o prediccions incompletes)
        df_valid = df_test_pred[[col_real, col]].dropna()
        y_true = df_valid[col_real].values
        y_pred = df_valid[col].values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        metriques[col] = [rmse, mse, mae]

    df_metriques = pd.DataFrame(metriques).set_index('Mètrica')

    return df_metriques.round(4)



def plot_prediccions(
    df_train,
    df_val,
    df_test_pred,
    col_preds=['pred_batch', 'pred_iter', 'pred_reinject'],
    dies_train=0,
    mostrar_val=False,
    title='Temperatura real i predicció LSTM',
    show=True
):
    """
    Genera una figura amb les dades reals i les prediccions d'un model LSTM per a una estació meteorològica.

    Aquesta funció permet representar les dades reals de la sèrie temporal de temperatura (train, validació i test),
    així com les prediccions generades pel model LSTM, amb colors fixos per cada estratègia de predicció per tal de mantenir la consistència visual.

    Args:
        df_train (pd.DataFrame): DataFrame amb les dades d'entrenament. Ha de contenir com a mínim ['data', 'valor'].
        df_val (pd.DataFrame): DataFrame amb les dades de validació. Ha de contenir ['data', 'valor'].
        df_test_pred (pd.DataFrame): DataFrame amb les dades de test i prediccions. Ha de contenir ['data', 'valor'] i les columnes de predicció.
        col_preds (list of str): Llista de noms de columnes de predicció a representar, com ara ['prediccio_batch', 'prediccio_iter'].
        dies_train (int): Nombre de dies finals del train que es volen mostrar (només si mostrar_train=True).
        mostrar_val (bool): Si es vol representar la sèrie de validació.
        title (str): Títol del gràfic.
        station (str): Nom de l'estació per afegir al títol.
        show (bool): Si es vol mostrar el gràfic al final de la funció. Per defecte True.

    Returns:
        fig (matplotlib.figure.Figure): Objecte figura amb el gràfic generat.
    """

    fig, ax = plt.subplots(figsize=(16, 5))

    # Mostrar últims dies del train i validació si s'especifica
    if mostrar_val:
        if dies_train > 0:
            data_limit = df_train['data'].max() - pd.Timedelta(days=dies_train)
            df_train_filtrat = df_train[df_train['data'] >= data_limit]
            ax.plot(df_train_filtrat['data'], df_train_filtrat['valor'], label=f'Train (últims {dies_train} dies)', color='firebrick', linewidth=1.5)

        ax.plot(df_val['data'], df_val['valor'], label='Validació', color='darkgreen', linewidth=1.5)


    # Test (color fix)
    ax.plot(df_test_pred['data'], df_test_pred['valor'],
            label='Test', color='steelblue', linewidth=1.5)

    # Colors fixos per a cada estratègia coneguda
    colors_pred = {
        'pred_batch': 'darkorange',
        'pred_iter': 'purple',
        'pred_reinject': 'green'
    }

    linestyle_pred = '--'

    # Prediccions
    if col_preds:
        
        for col in col_preds:
            
            if col not in df_test_pred.columns:
                continue  # Si la columna no existeix, la saltem
            
            color = colors_pred.get(col, 'gray')  # Color per defecte si no està definit
            label = col.replace('_', ' ').capitalize()
            ax.plot(df_test_pred['data'], df_test_pred[col],
                    label=label, color=color, linestyle=linestyle_pred, linewidth=1.5)


    # Format general del gràfic
    ax.set_title(f'{title}', fontsize=17, weight='bold')
    ax.set_xlabel('Data', fontsize=14)
    ax.set_ylabel('Temperatura (°C)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.6)
    ax.legend(fontsize=12, frameon=False)

    # Format de les dates a l'eix X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate()
    fig.tight_layout()

    # Mostrar el gràfic si s'indica
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig




# ============================================================================================
# Funcions unificades per a la creació i entrenament del model LSTM i prediccions
# Aquestes funcions encapsulen tot el procés de preparació de dades, entrenament i predicció
# =============================================================================================



# Funció principal per Definir i entrenar el model LSTM

def deftrain_model_lstm(
    df_lstm,                 # DataFrame amb la columna 'valor' a escalar i utilitzar
    window_size=24,          # Mida de la finestra temporal (timesteps)
    n_outputs=1,             # Nombre de passos a predir (1 per regressió simple, >1 per multi-output)
    lookahead=0,             # Passos entre el final de la finestra i la primera predicció (per defecte 0)
    n_layers=3,              # Nombre de capes LSTM
    n_units=64,              # Nombre de neurones per capa LSTM
    dropout_rate=0.2,        # Percentatge de dropout entre capes
    optimizer='adam',        # Optimitzador per compilar el model
    loss='mse',              # Funció de pèrdua per compilar el model
    epochs=50,               # Nombre d'èpoques d'entrenament
    batch_size=32,           # Mida del lot per l'entrenament
    patience=5,              # Paciencia per EarlyStopping
    shuffle=False,           # Si es barregen les dades durant l'entrenament (sempre False en seqüències temporals)
    seed=42,                 # Llavor per a la reproduïbilitat (per fixar llavors aleatòries en numpy i tensorflow)
    summary=True,            # Si es vol mostrar el resum del model al final de l'entrenament
    show=True                # Si es vol mostrar el gràfic de pèrdua d'entrenament i validació
):
    """
    Entrena un model LSTM amb les dades proporcionades, aplicant escalat i creació de seqüències.

    Args:
        df_lstm (pd.DataFrame): DataFrame amb la columna 'valor'.
        window_size (int): Mida de la finestra temporal.
        n_outputs (int): Nombre de passos a predir.
        n_layers (int): Nombre de capes LSTM.
        n_units (int): Nombre de neurones per capa.
        dropout_rate (float): Percentatge de dropout entre capes.
        optimizer (str): Optimitzador.
        loss (str): Funció de pèrdua.
        epochs (int): Nombre d’èpoques d’entrenament.
        batch_size (int): Mida del lot.
        patience (int): Paciencia per EarlyStopping.
        shuffle (bool): Si es barregen les dades.
        seed (int): Sement per a la reproduïbilitat.

    Returns:
        model, scaler, X_train, y_train, X_val, y_val, X_test, y_test,
        df_train, df_val, df_test, history
    """
    # Separar el DataFrame en train, val i test
    df_train, df_val, df_test = split_dades(df_lstm)

    # Escalar les dades
    df_train, df_val, df_test, scaler = escalar_dades(df_train, df_val, df_test)


    # Crear seqüències per la LSTM
    X_train, y_train = create_sequences(df_train['valor_scaled'].values, window_size=window_size, n_outputs=n_outputs, lookahead=lookahead, n_slide=n_outputs)
    X_val, y_val = create_sequences(df_val['valor_scaled'].values, window_size=window_size, n_outputs=n_outputs, lookahead=lookahead ,n_slide=n_outputs)
    X_test, y_test = create_sequences(df_test['valor_scaled'].values, window_size=window_size, n_outputs=n_outputs, lookahead=lookahead, n_slide=n_outputs)



    # Definir i compilar el model LSTM
    model = definir_model_lstm(window_size, n_features=1, n_outputs=n_outputs,
                               n_layers=n_layers, n_units=n_units,
                               dropout_rate=dropout_rate,
                               optimizer=optimizer, loss=loss)


    # Entrenar el model
    history = train_model(model, X_train, y_train, X_val, y_val,
                          epochs=epochs, batch_size=batch_size,
                          patience=patience, shuffle=shuffle, seed=seed,summary=summary)



    # Mostrar Gràfic de pèrdua d'entrenament i validació
    fig_loss_train = plot_loss_train_val(history, show=show)


    # Retornar els objectes claus del procés
    print('Entrenament completat.')
    return model, scaler, X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, history, fig_loss_train



# Funció per aplicar prediccions amb un model LSTM entrenat

def prediu_model_lstm(
    model,                                                      # Model LSTM entrenat
    X_test,                                                     # Seqüències d’entrada per a test (forma: (n_samples, window_size, 1))
    df_test,                                                    # DataFrame original de test amb la columna 'valor_scaled'
    scaler,                                                     # MinMaxScaler utilitzat per desescalar les prediccions
    window_size,                                                # Mida de la finestra temporal
    n_outputs ,                                                 # Nombre de passos de predicció (1 per regressió simple, >1 per multi-output)
    lookahead=0,                                               # Passos entre el final de la finestra i la primera predicció (per defecte 0)
    met_pred = ['pred_batch', 'pred_iter', 'pred_reinject']     # Metodes de predicció a utilitzar
):
    """
    Aplica prediccions amb un model LSTM entrenat.

    Args:
        model (keras.Model): Model LSTM entrenat.
        X_test (np.array): Seqüències d’entrada per a test.
        df_test (pd.DataFrame): DataFrame original de test.
        scaler (MinMaxScaler): Escalador utilitzat per desescalar.
        window_size (int): Mida de finestra temporal.
        n_outputs (int): Nombre de passos de predicció.
        lookahead (int): Passos entre el final de la finestra i la primera predicció.
        met_pred (list): Llista de mètodes de predicció a utilitzar. Per defecte, inclou 'pred_batch', 'pred_iter' i 'pred_reinject'.

    Returns:
        df_test_pred (pd.DataFrame): Test amb prediccions.
        metriques (dict): Diccionari amb mètriques d’error.
    """


    # Crear copia df_test per a les prediccions 
    df_test_pred = df_test.copy()


    if n_outputs == 1:

        if 'pred_batch' in met_pred:
            print("Fent predicció batch...")
            df_test_pred = prediccio_batch(model, X_test, df_test_pred, scaler,lookahead=lookahead)
        
        if 'pred_iter' in met_pred:
            print("Fent predicció iterativa...")
            df_test_pred = prediccio_step_iterativa(model, X_test, df_test_pred, scaler,lookahead=lookahead)

        if 'pred_reinject' in met_pred:
            print("Fent predicció iterativa amb reinjecció...")
            df_test_pred = prediccio_iterativa_reinjection(model, X_test, df_test_pred, scaler,lookahead=lookahead)


    else:
        print("Fent predicció batch multi-output...")
        df_test_pred = prediccio_batch_multi(model, X_test, df_test_pred, scaler,
                                             window_size=window_size, 
                                             n_outputs=n_outputs,
                                             lookahead=lookahead)

        
   
    # Calcular mètriques per a les prediccions
    metriques = calcular_metriques(df_test_pred, col_real='valor',
                                    col_preds=met_pred)
        

    # Retornar el DataFrame de test amb les prediccions i el dataframe de mètriques
    return df_test_pred, metriques




# ============================================================================================================
# Pipeline que unifica les funcions de creació, entrenament i predicció, amb opcions de plot i guardat
# ============================================================================================================

def pipeline_lstm(
    df_lstm,
    window_size=24,
    n_outputs=1,
    lookahead=0,
    n_layers=3,
    n_units=64,
    dropout_rate=0.2,
    optimizer='adam',
    loss='mse',
    epochs=50,
    batch_size=32,
    patience=5,
    shuffle=False,
    seed=42,
    dies_train=0,
    mostrar_val=False,
    col_preds=['pred_batch', 'pred_iter', 'pred_reinject'],
    save_path=None,
    show=True,
    summary=True
):
    """
    Pipeline complet per entrenar, predir i visualitzar un model LSTM de predicció de temperatura.

    Aquesta funció integra els tres blocs principals del flux de treball amb xarxes LSTM:
    - Entrenament del model amb dades seqüencials
    - Predicció sobre el conjunt de test
    - Visualització de les prediccions juntament amb les dades reals

    Si s'indica un directori a `save_path`, es guardaran automàticament:
        - El model entrenat (`model.h5`)
        - L’historial d'entrenament (`loss_history.csv`) i la gràfica de pèrdua (`loss_plot.png`)
        - Les prediccions (`prediccions.csv`)
        - Les mètriques d’avaluació (`metrics.csv` i `metrics.txt`)
        - La configuració de l’experiment (`config.json`)

    Retorna:
        model, scaler, df_train, df_val, df_test_pred, history, metriques, fig, fig_loss_train
    """

    print("🧠 [1/5] Entrenant el model LSTM...")

    model, scaler, X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, history, fig_loss_train = deftrain_model_lstm(
        df_lstm=df_lstm,
        window_size=window_size,
        n_outputs=n_outputs,
        lookahead=lookahead,
        n_layers=n_layers,
        n_units=n_units,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
        loss=loss,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        shuffle=shuffle,
        seed=seed,
        summary= summary,
        show=show
    )

    if save_path:
        print("💾 [2/5] Guardant model i gràfica de pèrdua...")
        os.makedirs(save_path, exist_ok=True)
        model.save(os.path.join(save_path, "model.h5"), include_optimizer=False)
        pd.DataFrame(history.history).to_csv(os.path.join(save_path, "loss_history.csv"))
        fig_loss_train.savefig(os.path.join(save_path, "loss_plot.png"))

    else:
        print("📂 [2/5] Model entrenat")

    print("🔮 [3/5] Fent prediccions...")

    df_test_pred, metriques = prediu_model_lstm(
        model=model,
        X_test=X_test,
        df_test=df_test,
        scaler=scaler,
        window_size=window_size,
        n_outputs=n_outputs,
        lookahead=lookahead,
        met_pred=col_preds
    )

    print("📊 [4/5] Generant gràfic de prediccions...")

    fig = plot_prediccions(
        df_train=df_train,
        df_val=df_val,
        df_test_pred=df_test_pred,
        col_preds=col_preds,
        dies_train=dies_train,
        mostrar_val=mostrar_val,
        title='Temperatura real i predicció LSTM',
        show=show
    )

    # Mostrar les metriques calculades
    print("\n📈 Mètriques calculades:")
    print(metriques)


    if save_path:
        print("🗃️ [5/5] Guardant prediccions, mètriques i configuració...")
        
        df_test_pred.to_csv(os.path.join(save_path, "prediccions.csv"), index=False)
        metriques.to_csv(os.path.join(save_path, "metrics.csv"))

        # Guardar les mètriques com si fos un CSV però en fitxer .txt
        metriques.to_csv(os.path.join(save_path, "metrics.txt"))


        fig.savefig(os.path.join(save_path, "plot.png"))

        config = {
            'window_size': window_size,
            'n_outputs': n_outputs,
            'lookahead': lookahead,
            'n_layers': n_layers,
            'n_units': n_units,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer,
            'loss': loss,
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'shuffle': shuffle,
            'seed': seed
        }

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    else:
        print("📂 [5/5] Pipeline completada sense guardar resultats.")

    return model, scaler, df_train, df_val, df_test_pred, history, metriques, fig, fig_loss_train




# ===============================================================
# FUNCIÓ PER CONSTRUIR NOM D'EXPERIMENT
# ===============================================================


def construir_nom_experiment(params: dict, prefix="exp"):
    """
    Construeix un nom únic per identificar cada experiment segons els hiperparàmetres.
    """
    parts = [
        f"win{params.get('window_size', 24)}",
        f"out{params.get('n_outputs', 1)}",
        f"look{params.get('lookahead', 0)}",
        f"lay{params.get('n_layers', 2)}",
        f"uni{params.get('n_units', 64)}",
        f"drop{int(params.get('dropout_rate', 0.2) * 100)}"
    ]
    nom = "_".join(parts)
    return f"{prefix}_{nom}"






# ===============================================================
# WRAPPERS D'ENTRENAMENT I EXECUCIÓ D'EXPERIMENTS
# ===============================================================


def executar_experiment(
    df,
    params: dict,
    save_path: str = None,
    col_preds=['pred_batch', 'pred_iter', 'pred_reinject'],
    dies_train=0,
    mostrar_val=False
):
    """
    Wrapper per executar un experiment amb la pipeline LSTM i guardar-ne els resultats.

    Args:
        df (pd.DataFrame): DataFrame amb les dades originals.
        params (dict): Diccionari amb hiperparàmetres de l'experiment.
        save_path (str, optional): Ruta per guardar els resultats.
        col_preds (list): Columnes de predicció a mostrar o guardar.
        dies_train (int): Dies finals de train a mostrar (si escau).
        mostrar_val (bool): Si s'ha de mostrar també la validació.
    """


    if save_path is None:
        nom = construir_nom_experiment(params)
        save_path = os.path.join("resultats", nom)

    pipeline_lstm(
        df_lstm=df,
        window_size=params.get('window_size', 24),
        n_outputs=params.get('n_outputs', 1),
        lookahead=params.get('lookahead', 0),
        n_layers=params.get('n_layers', 3),
        n_units=params.get('n_units', 64),
        dropout_rate=params.get('dropout_rate', 0.2),
        optimizer=params.get('optimizer', 'adam'),
        loss=params.get('loss', 'mse'),
        epochs=params.get('epochs', 50),
        batch_size=params.get('batch_size', 32),
        patience=params.get('patience', 5),
        shuffle=params.get('shuffle', False),
        seed=params.get('seed', 42),
        dies_train=dies_train,
        mostrar_val=mostrar_val,
        col_preds=col_preds,
        save_path=save_path,
        summary=False,         # ❌ Desactiva el resum del model al fer experiments massius
        show=False             # ❌ Desactiva el plot interactiu al fer experiments massius
    )































