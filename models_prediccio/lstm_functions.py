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



# Creem una funció per crear sequences per LSTM d'entrada

def create_sequences(series, window_size, n_outputs=1, n_slide=1):
    """
    Crea seqüències d'entrada i sortida per predicció simple o multi-output, 
    amb control del pas de desplaçament entre finestres.

    Args:
        series (array): sèrie temporal escalada.
        window_size (int): llargada de la finestra d'entrada.
        n_outputs (int): nombre de passos a predir (per defecte 1).
        n_slide (int): quant avancem la finestra a cada iteració (per defecte 1).

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
    X, y = [], []
    i = window_size
    while i <= len(series) - n_outputs:
        X.append(series[i - window_size:i])
        if n_outputs == 1:
            y.append(series[i])
        else:
            y.append(series[i:i + n_outputs])
        i += n_slide
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    if n_outputs == 1:
        y = y.reshape(-1, 1)  # Ara fa el reshape que Keras espera per a regressió 1D
        
    return X, y


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
        display(df_train.head())

        print('Validation dataset shape:', df_val.shape)
        display(df_val.head())

        print('Test dataset shape:', df_test.shape)
        display(df_test.head())

    return df_train, df_val, df_test, scaler



# ================================================================
# Funcions per a la creació i entrenament del model LSTM
# ================================================================



def definir_model_lstm(
    window_size,       # mida de la finestra temporal (timesteps)
    n_features,        # nombre de variables d'entrada (features)
    n_outputs=1,       # nombre de valors a predir
    n_layers=3,        # nombre total de capes LSTM
    n_units=64,        # neurones per cada capa LSTM
    dropout_rate=0.2,  # percentatge de neurones a desactivar (Dropout)
    usar_dropout=True, # afegir Dropout entre capes
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
        usar_dropout (bool): si es vol afegir Dropout entre capes.
        optimizer (str): optimitzador per compilar el model.
        loss (str): funció de pèrdua per compilar el model.

    Returns:
        model (keras.Sequential): model LSTM ja compilat.
    """
    model = Sequential()

    # Capa inicial LSTM (amb input_shape)
    # Si hi ha més d'una capa, cal que retorni seqüència
    model.add(LSTM(n_units, return_sequences=(n_layers > 1), input_shape=(window_size, n_features)))
    if usar_dropout:
        model.add(Dropout(dropout_rate))

    # Capes intermèdies (si n_layers >= 3)
    for _ in range(n_layers - 2):
        model.add(LSTM(n_units, return_sequences=True))
        if usar_dropout:
            model.add(Dropout(dropout_rate))

    # Última capa LSTM (si n_layers >= 2)(sense return_sequences)
    if n_layers > 1:
        model.add(LSTM(n_units))
        if usar_dropout:
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
    summary=False       # si es vol mostrar el resum del model al final
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




def plot_loss_train_val(history):

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
    plt.show()




# ================================================================
# Funcions de predicció LSTM 1 output
# ================================================================

def prediccio_batch(model, X_test, df_test_pred, scaler, nom_columna='pred_batch'):
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
    idx_valid = df_test_pred.index[window_size:]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred




def prediccio_step_iterativa(model, X_test, df_test_pred, scaler, nom_columna='pred_iter'):
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
    idx_valid = df_test_pred.index[window_size:window_size + n_passos]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred



def prediccio_iterativa_reinjection(model, X_test, df_test_pred, scaler, reinjeccio=5, nom_columna='pred_reinject'):
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
    idx_valid = df_test_pred.index[window_size : window_size + n_preds]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred




# ================================================================
# Funcions de predicció LSTM multi-output
# ================================================================


# Funció per fer prediccions multi-step

def prediccio_batch_multi(model, X_test, df_test, scaler, window_size, n_outputs, nom_columna='pred_batch'):

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

    Retorna:
        df_test amb la nova columna `nom_columna` que conté les prediccions (amb NaNs on no es pot predir).
    """
    # 1. Fer la predicció batch
    y_pred = model.predict(X_test, verbose=0)

    # 2. Desescalar les prediccions (per tornar a °C)
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # 3. Inicialitzem la nova columna amb NaNs
    df_test[nom_columna] = np.nan

    # 4. Omplim la columna amb les prediccions multi-output (una fila per cada pas predit)
    for i in range(len(y_pred_rescaled)):
        for j in range(n_outputs):
            idx = window_size + i * n_outputs + j  # índex corresponent a la predicció j de la i-èsima seqüència
            if idx < len(df_test):
                df_test.at[idx, nom_columna] = y_pred_rescaled[i, j]

    return df_test




def prediccio_step_iterativa_multi(model, X_test, df_test_pred, scaler, nom_columna='pred_iter'):
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

    Returns:
        df_test_pred (DataFrame): amb la nova columna de predicció iterativa afegida.
        y_pred_rescaled (np.array): prediccions desescalades (forma: (n_preds_total,)).
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
    idx_valid = df_test_pred.index[window_size:window_size + len(y_pred_rescaled)]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred




def prediccio_iterativa_reinjection_multi(model, X_test, df_test_pred, scaler, reinjeccio=5, nom_columna='pred_reinject'):
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
        y_pred_rescaled (np.array): prediccions desescalades (forma: (n_preds_total,)).
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
    idx_valid = df_test_pred.index[window_size : window_size + len(y_pred_rescaled)]
    df_test_pred.loc[idx_valid, nom_columna] = y_pred_rescaled

    return df_test_pred



# ================================================================
# Funcions de càlcul de mètriques i gràfics
# ================================================================


def calcular_metriques(df_test_pred, col_real='valor', col_preds=['pred_batch', 'pred_iter', 'pred_reinject'],window_size=0):
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
    y_true = df_test_pred[col_real][window_size:]

    for col in col_preds:
        y_pred = df_test_pred[col][window_size:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        metriques[col] = [rmse, mse, mae]

    df_metriques = pd.DataFrame(metriques).set_index('Mètrica').round(4)
    
    return df_metriques




def calcular_metriques_multiout(df_test_pred, col_real='valor', col_preds=['pred_batch', 'pred_iter', 'pred_reinject'],window_size=0, n_outputs=1):
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
    y_true = df_test_pred[col_real][window_size:(1-n_outputs)]

    for col in col_preds:
        y_pred = df_test_pred[col][window_size:(1-n_outputs)]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        metriques[col] = [rmse, mse, mae]

    df_metriques = pd.DataFrame(metriques).set_index('Mètrica').round(4)
    
    return df_metriques




def plot_prediccions_lstm(
    df_train,
    df_val,
    df_test_pred,
    columnes_prediccio=None,
    mostrar_train=False,
    dies_train=0,
    mostrar_val=False,
    title='Temperatura real i predicció LSTM',
    station='Z1 de la Bonaigua'
):
    
    """
    Ploteja les dades reals i les prediccions d'un model LSTM.

    Args:
        df_train (pd.DataFrame): DataFrame de train amb columnes ['data', 'valor'].
        df_val (pd.DataFrame): DataFrame de validació amb ['data', 'valor'].
        df_test_pred (pd.DataFrame): DataFrame de test amb ['data', 'valor'] i prediccions.
        columnes_prediccio (list): Columnes de predicció a representar (ex: ['prediccio_batch', 'prediccio_iter']).
        mostrar_train (bool): Si es vol mostrar part del train.
        dies_train (int): Dies finals de train a mostrar (si mostrar_train=True).
        mostrar_val (bool): Si es vol mostrar validació.
        title (str): Títol del gràfic.
        station (str): Nom de l'estació per incloure al títol.
    """
    plt.figure(figsize=(16, 5))

    if mostrar_train and dies_train > 0:
        data_limit = df_train['data'].max() - pd.Timedelta(days=dies_train)
        df_train_filtrat = df_train[df_train['data'] >= data_limit]
        plt.plot(df_train_filtrat['data'], df_train_filtrat['valor'],
                 label=f'Train (últims {dies_train} dies)', color='firebrick', linewidth=1.5)

    if mostrar_val:
        plt.plot(df_val['data'], df_val['valor'],
                 label='Validació', color='darkgreen', linewidth=1.5)

    # Dades reals de test
    plt.plot(df_test_pred['data'], df_test_pred['valor'],
             label='Test', color='steelblue', linewidth=1.5)

    # Prediccions
    colors = ['darkorange', 'purple', 'green', 'brown', 'teal']
    linestyles = ['--', '--', '--', '--', '--']

    if columnes_prediccio:
        for i, col in enumerate(columnes_prediccio):
            plt.plot(df_test_pred['data'], df_test_pred[col],
                     label=col.replace('_', ' ').capitalize(),
                     color=colors[i % len(colors)],
                     linestyle=linestyles[i % len(linestyles)],
                     linewidth=1.5)

    # Format general
    plt.title(f'{title} - Estació {station}', fontsize=17, weight='bold')
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Temperatura (°C)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.6)
    plt.legend(fontsize=12, frameon=False)

    # Format de dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y %m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()
