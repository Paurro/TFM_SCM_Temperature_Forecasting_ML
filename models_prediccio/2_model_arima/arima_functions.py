# """
# ===============================================================================
#  📦 arima_functions.py
# ===============================================================================
#  Llibreria de funcions modulars per a models ARIMA/SARIMA/SARIMAX aplicats
#  a predicció de sèries temporals amb dades horàries.

#  Desenvolupat per: [El teu nom o equip]
#  Data: 2025-06-25

#  Funcionalitat principal:
#  - Entrenament i predicció amb models ARIMA/SARIMA
#  - Rolling forecast amb reinjecció (multi-pas per bloc)
#  - Càlcul de mètriques d’error (MAE, RMSE)
#  - Gràfics de predicció amb context històric
#  - Pipelines automàtiques amb opcions de guardat
#  - Suport per experiments i comparatives

#  Llibreries necessàries:
#  - numpy
#  - pandas
#  - matplotlib
#  - statsmodels
#  - scikit-learn
#  - tqdm

#  Estructura principal:
#  - split_train_test_arima(...)
#  - entrenar_model_arima(...)
#  - prediccio_arima(...)
#  - calcular_metriques_arima(...)
#  - plot_prediccions_arima(...)
#  - pipeline_arima(...)
#  - pipeline_rolling_forecast_arima(...)

#  Ús recomanat:
#  - Importar el fitxer des d’un notebook o script principal
#  - Definir paràmetres i cridar les pipelines amb verbose i save_path

# ===============================================================================
# """

# ===============================================================================
# Importem les llibreries necessàries
# ===============================================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error



# ===============================================================================
# Funcions per dividir el DataFrame en conjunts d'entrenament i test
# ===============================================================================

def split_train_test_arima(
    df,
    data_inici_pred,
    dies_entrenament=60,
    data_final_pred=None,
    verbose=False
):
    """
    Divideix el DataFrame en df_train i df_test per ARIMA,
    amb una finestra d'entrenament retrospectiva.

    Parameters:
        df (pd.DataFrame): DataFrame amb columnes 'data' i 'valor'
        data_inici_pred (str o datetime): inici del període de predicció
        dies_entrenament (int): nombre de dies enrere per entrenar (default = 60)
        data_final_pred (str o datetime, optional): límit final del test
        verbose (bool): mostrar rangs i longituds dels conjunts

    Returns:
        df_train, df_test (pd.DataFrames): conjunts d'entrenament i test
    """
    data_inici_pred = pd.to_datetime(data_inici_pred, utc=True)
    data_inici_train = data_inici_pred - pd.Timedelta(days=dies_entrenament)
    data_inici_train = pd.to_datetime(data_inici_train, utc=True)

    df_train = df[(df['data'] >= data_inici_train) & (df['data'] < data_inici_pred)].copy()
    df_train = df_train.reset_index(drop=True)  # Restablir l'index

    if data_final_pred:
        data_final_pred = pd.to_datetime(data_final_pred,utc=True)
        df_test = df[(df['data'] >= data_inici_pred) & (df['data'] <= data_final_pred)].copy()
    else:
        df_test = df[df['data'] >= data_inici_pred].copy()

    df_test = df_test.reset_index(drop=True)  # Restablir l'index

    if verbose:
        print("🔹 Split ARIMA:")
        print(f"  ▸ Train: {len(df_train)} valors")
        print(f"  ▸ Test:  {len(df_test)} valors")

    return df_train, df_test


# ===============================================================================
# Funcions per entrenar el model ARIMA/SARIMA i fer prediccions
# ===============================================================================


def entrenar_model_arima(
    df_train,
    p, d, q,
    P=None, D=None, Q=None, s=None,
    verbose=False
):
    """
    Entrena un model ARIMA o SARIMA amb paràmetres separats.

    Parameters:
        df_train (pd.DataFrame): conjunt d'entrenament amb columna 'valor'
        p, d, q (int): Paràmetres ARIMA
        P, D, Q, s (int or None): Paràmetres SARIMA. Si es donen, s'utilitza component estacional.
        verbose (bool): Si True, imprimeix informació de l'entrenament

    Returns:
        model_fit: model SARIMAX entrenat
    """
    
    order = (p, d, q)
    use_seasonal = None not in (P, D, Q, s)

    if use_seasonal:
        seasonal_order = (P, D, Q, s)
        if verbose:
            print(f"📦 Entrenant SARIMA({p},{d},{q}) x {seasonal_order}")
        model = SARIMAX(df_train['valor'], order=order, seasonal_order=seasonal_order)
    else:
        if verbose:
            print(f"📦 Entrenant ARIMA({p},{d},{q}) (sense component estacional)")
        model = SARIMAX(df_train['valor'], order=order)

    model_fit = model.fit()

    if verbose:
        print(model_fit.summary())

    return model_fit



def prediccio_arima(df_test, model_fit, n_passos=None,verbose=False):
    """
    Fa la predicció amb un model ARIMA/SARIMA ja entrenat a partir del df_test.

    Parameters:
        df_test (pd.DataFrame): DataFrame amb columna 'valor' i 'data'
        model_fit: model entrenat
        n_passos (int or None): nombre de passos a predir (en aquest cas hores). Si None, prediu tot df_test.

    Returns:
        df_test_pred (pd.DataFrame): df_test amb la columna 'forecast' afegida
    """
    df_pred = df_test.copy().reset_index(drop=True)
    if n_passos is None:
        n_passos = len(df_pred)

    forecast = model_fit.forecast(steps=n_passos)
    forecast.reset_index(drop=True, inplace=True)
    df_pred['forecast'] = forecast
    

    if verbose:
        print("🔹 Predicció ARIMA:")
        print(f"  ▸ Passos predits: {n_passos}")
        print(df_pred.head(5))

    return df_pred



# ===============================================================================
# Funcions per calcular mètriques i generar gràfics de predicció
# ===============================================================================

def calcular_metriques_arima(df_pred, col_real='valor', col_pred='forecast', verbose=True):
    """
    Calcula les mètriques d'avaluació (MAE i RMSE) entre la columna real i la predicció.

    Parameters:
        df_pred (pd.DataFrame): DataFrame amb columnes de valors reals i predits
        col_real (str): nom de la columna de valors reals
        col_pred (str): nom de la columna de predicció
        verbose (bool): si True, imprimeix les mètriques

    Returns:
        dict: {'mae': float, 'rmse': float}
    """
    df = df_pred[[col_real, col_pred]].dropna()
    y_true = df[col_real].values
    y_pred = df[col_pred].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)


    metriques = pd.DataFrame({
        'RMSE': [rmse],
        'MSE': [mse],
        'MAE': [mae]
    })

    if verbose:
        print(f"📏 RMSE: {rmse:.2f} °C")
        print(f"📏 MSE: {mse:.2f} °C²")
        print(f"📏 MAE: {mae:.2f} °C")

    return metriques



def plot_prediccions_arima(
    df_pred,
    df_train=None,
    plot_dies_ant=0,
    title="Predicció de temperatura amb ARIMA",
    xlabel="Data",
    ylabel="Temperatura (°C)",
    show=True
):
    """
    Dibuixa la predicció de temperatura feta amb ARIMA.

    Parameters:
        df_pred (pd.DataFrame): ha de contenir 'data', 'valor' i 'forecast'
        df_train (pd.DataFrame, optional): opcional, context d'entrenament
        plot_dies_ant (int): dies previs del train a mostrar
        title, xlabel, ylabel (str): elements visuals
        show (bool): si True mostra el gràfic; sinó retorna fig
    Returns:
        fig (matplotlib.figure.Figure)
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Validar df_train i assegurar que conté les columnes esperades
    if df_train is not None and isinstance(df_train, pd.DataFrame):
        if all(col in df_train.columns for col in ['data', 'valor']):
            if plot_dies_ant > 0:
                data_inici_plot = df_pred['data'].min() - pd.Timedelta(days=plot_dies_ant)
                df_train_plot = df_train[df_train['data'] >= data_inici_plot]
                ax.plot(df_train_plot['data'], df_train_plot['valor'], label='Train', color='tab:gray')

    # Plot de la sèrie real i la predicció
    ax.plot(df_pred['data'], df_pred['valor'], label='Real', color='tab:blue')
    ax.plot(df_pred['data'], df_pred['forecast'], label='Forecast', color='tab:orange', linestyle='--')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if show:
        plt.show()
    return fig





# ===============================================================================
# Funcions per a pipelines completes d'ARIMA/SARIMA
# ===============================================================================

def pipeline_arima(
    df,
    data_inici_pred,
    data_final_pred=None,
    dies_entrenament=60,
    p=1, d=1, q=1,
    P=None, D=None, Q=None, s=None,
    n_passos=None,
    plot_dies_ant=0,
    verbose=True,
    save_path=None,
    show=True
):
    """
    Pipeline completa d'ARIMA/SARIMA, estil modular i exportable.

    Parameters:
        df (pd.DataFrame): dades amb columnes 'data' i 'valor'
        data_inici_pred (str): moment a partir del qual es fa la predicció
        data_final_pred (str): límit del període a predir (si None, fins al final)
        dies_entrenament (int): finestra retrospectiva pel train
        p,d,q,P,D,Q,s: paràmetres ARIMA/SARIMA
        n_passos (int or None): nombre d'hores a predir. Si None, es calcula a partir de data_final_pred.
        plot_dies_ant (int): dies de context anterior per mostrar al gràfic
        verbose (bool): si True, imprimeix els passos
        save_path (str or None): si s’indica, guarda tots els resultats
        show (bool): si True, mostra els gràfics

    Returns:
        model_fit, df_train, df_pred, metrics, fig
    """

    if verbose: print("🧠 [1/5] Preparant les dades...")
    df_train, df_test = split_train_test_arima(
        df,
        data_inici_pred=data_inici_pred,
        data_final_pred=data_final_pred,
        dies_entrenament=dies_entrenament,
        verbose=verbose
    )

    if verbose: print("⚙️ [2/5] Entrenant el model ARIMA...")
    model_fit = entrenar_model_arima(
        df_train,
        p=p, d=d, q=q,
        P=P, D=D, Q=Q, s=s,
        verbose=verbose
    )

    if verbose: print("🔮 [3/5] Fent la predicció...")

    
    # Determinar n_passos en funció de data_final_pred i n_passos

    if data_final_pred is not None:
        # Si s'ha indicat data_final_pred, fem-la prevaldre
        n_passos_calc = int((pd.to_datetime(data_final_pred) - pd.to_datetime(data_inici_pred)) / pd.Timedelta(hours=1)) + 1
        if verbose:
            print(f"ℹ️ [3] 'data_final_pred' indicat. Calculant n_passos = {n_passos_calc}")
        n_passos = n_passos_calc

    elif n_passos is None:
        # Si no es dona ni data_final_pred ni n_passos, fem servir tot el test
        n_passos = len(df_test)
        if verbose:
            print(f"ℹ️ [3] Ni 'n_passos' ni 'data_final_pred' indicats. Predirem tot el test: {n_passos} passos.")

    # ✅ Assegurar que n_passos no excedeix la mida del test
    max_passos = len(df_test)
    if n_passos > max_passos:
        if verbose:
            print(f"⚠️ [3] 'n_passos' ({n_passos}) excedeix la mida del test ({max_passos}). S'ajustarà.")
        n_passos = max_passos


    df_pred = prediccio_arima(df_test, model_fit=model_fit,n_passos=n_passos, verbose=verbose)

    if verbose: print("📏 [4/5] Calculant mètriques...")
    metrics = calcular_metriques_arima(df_pred,col_real='valor', col_pred='forecast', verbose=verbose)

    if verbose: print("📊 [5/5] Generant gràfic de predicció...")
    fig = plot_prediccions_arima(
        df_pred,
        df_train=df_train,
        plot_dies_ant=plot_dies_ant,
        show=show
    )

    if save_path:
        if verbose: print("💾 Guardant resultats a disc...")
        os.makedirs(save_path, exist_ok=True)
      
         # Guardar el model entrenat
        model_fit.save(os.path.join(save_path, "model.pkl"))
        
        # Guardar els DataFrames i mètriques
        df_pred.to_csv(os.path.join(save_path, "prediccions.csv"), index=False)
        df_train.to_csv(os.path.join(save_path, "train.csv"), index=False)
        
        # Guardar les mètriques en format CSV i TXT
        metrics.to_csv(os.path.join(save_path, "metrics.csv"), index=False)
        metrics.to_csv(os.path.join(save_path, "metrics.txt"), index=False)
        

        # Guardar el gràfic
        fig.savefig(os.path.join(save_path, "plot.png"))
        
        config = {
            'data_inici_pred': str(data_inici_pred),
            'data_final_pred': str(data_final_pred),
            'dies_entrenament': dies_entrenament,
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q, 's': s
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
    else:
        if verbose: print("📂 Pipeline completada sense guardar resultats.")

    return model_fit, df_train, df_pred, metrics, fig




def pipeline_rolling_forecast_arima(
    df,
    data_inici_pred,
    data_final_pred=None,
    dies_entrenament=60,
    p=1, d=1, q=1,
    P=None, D=None, Q=None, s=None,
    n_passos=None,
    pas_pred=1,
    plot_dies_ant=0,
    verbose=True,
    save_path=None,
    show=True
):
    """
    Rolling forecast per blocs (re-injection) amb ARIMA/SARIMA.

    Parameters:
        df (pd.DataFrame): dades amb columnes 'data' i 'valor'
        data_inici_pred (str o datetime): inici de la predicció
        data_final_pred (str o datetime): límit de la predicció (alternativa a `n_passos`)
        dies_entrenament (int): finestra retrospectiva pel train
        p, d, q (int): paràmetres ARIMA
        P, D, Q, s (int or None): paràmetres SARIMA
        n_passos (int or None): nombre total d'hores a predir
        pas_pred (int): passos (hores) per reinjecció
        plot_dies_ant (int): dies de context anterior per mostrar al gràfic
        verbose (bool): si True, imprimeix els passos
        save_path (str or None): si s’indica, guarda resultats
        show (bool): si True, mostra els gràfics

    Returns:
        df_pred, metrics, fig
    """

    # ✅ Assegurar que les dates són timezone-aware
    data_inici_pred = pd.to_datetime(data_inici_pred)
    if data_inici_pred.tzinfo is None:
        data_inici_pred = data_inici_pred.tz_localize("UTC")

    if data_final_pred is not None:
        data_final_pred = pd.to_datetime(data_final_pred)
        if data_final_pred.tzinfo is None:
            data_final_pred = data_final_pred.tz_localize("UTC")
        n_passos = int((data_final_pred - data_inici_pred) / pd.Timedelta(hours=1)) + 1
        if verbose:
            print(f"ℹ️ [0] 'data_final_pred' indicat. Calculant n_passos = {n_passos}")
    elif n_passos is None:
        data_final = df['data'].max()
        n_passos = int((data_final - data_inici_pred) / pd.Timedelta(hours=1)) + 1
        if verbose:
            print(f"ℹ️ [0] 'n_passos' no especificat. Predirem fins a {data_final} ({n_passos} hores).")

    # 🔒 Validació per no sortir del rang
    max_hores = int((df['data'].max() - data_inici_pred) / pd.Timedelta(hours=1))
    if n_passos > max_hores:
        if verbose:
            print(f"⚠️ 'n_passos' ({n_passos}) excedeix les dades disponibles ({max_hores}). S'ajustarà.")
        n_passos = max_hores

    # 🔢 Definir dates i df_pred buida
    dates_pred = pd.date_range(start=data_inici_pred, periods=n_passos, freq='h')
    df_pred = df[(df['data'] >= data_inici_pred) & (df['data'] < data_inici_pred + pd.Timedelta(hours=n_passos))].copy().reset_index(drop=True)
    df_pred['forecast'] = np.nan

    if verbose:
        print(f"🧠 [1/4] Iniciant rolling forecast en blocs de {pas_pred} hora/es...")

    # 🔁 Rolling reinjection loop
    forecast_idx = 0
    while forecast_idx < len(dates_pred):
        current_time = dates_pred[forecast_idx]
        steps = min(pas_pred, len(dates_pred) - forecast_idx)

        if verbose:
            print(f"🔁 Reinjecció des de {current_time} → {steps} hora/es")

        df_train, _ = split_train_test_arima(
            df,
            data_inici_pred=current_time,
            dies_entrenament=dies_entrenament,
            verbose=False
        )

        model = entrenar_model_arima(df_train, p, d, q, P, D, Q, s, verbose=False)
        forecast = model.forecast(steps=steps).values
        df_pred.loc[forecast_idx:forecast_idx+steps-1, 'forecast'] = forecast
        forecast_idx += steps

    if verbose:
        print("📏 [2/4] Calculant mètriques...")
    metrics = calcular_metriques_arima(df_pred, verbose=verbose)

    if verbose:
        print("📊 [3/4] Generant gràfic de predicció...")
    fig = plot_prediccions_arima(df_pred, df_train=df_train, plot_dies_ant=plot_dies_ant, show=show)

    if save_path:
        if verbose:
            print("💾 [4/4] Guardant resultats...")
        os.makedirs(save_path, exist_ok=True)
        df_pred.to_csv(os.path.join(save_path, "prediccions.csv"), index=False)
        metrics.to_csv(os.path.join(save_path, "metrics.csv"), index=False)
        metrics.to_csv(os.path.join(save_path, "metrics.txt"), index=False)

        fig.savefig(os.path.join(save_path, "plot.png"))

        config = {
            'data_inici_pred': str(data_inici_pred),
            'data_final_pred': str(data_final_pred) if data_final_pred else None,
            'n_passos': n_passos,
            'pas_pred': pas_pred,
            'dies_entrenament': dies_entrenament,
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q, 's': s
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    else:
        if verbose:
            print("📂 [4/4] Pipeline completada sense guardar resultats.")

    return df_pred, metrics, fig




# ===============================================================================
# Funcions per construir noms d'experiment ARIMA/SARIMA
# ===============================================================================
def construir_nom_experiment_arima(params: dict, prefix=""):
    """
    Construeix un nom d'experiment únic a partir dels hiperparàmetres ARIMA/SARIMA.
    Si s'especifica un prefix (com "A0", "B1"...), es col·loca al principi del nom.
    """
    parts = []

    if prefix:
        parts.append(str(prefix))

    # Paràmetres ARIMA
    p = params.get('p', 1)
    d = params.get('d', 1)
    q = params.get('q', 1)
    parts.append(f"p{p}d{d}q{q}")

    # Paràmetres SARIMA (si existeixen i tenen valor)
    P = params.get('P')
    D = params.get('D')
    Q = params.get('Q')
    s = params.get('s')

    if None not in (P, D, Q, s):
        parts.append(f"P{P}D{D}Q{Q}s{s}")

    # Rolling forecast
    if params.get('rolling', False):
        pas_pred = params.get('pas_pred', 1)
        parts.append(f"roll{pas_pred}")

    # Finestra d'entrenament
    tr = params.get('dies_entrenament', 60)
    parts.append(f"tr{tr}")

    return "_".join(parts)



# ===============================================================================
# WRAPPER per fer experiments amb ARIMA/SARIMA
# ===============================================================================

def executar_experiment_arima(
    df,
    params: dict,
    save_root: str = "resultats_arima"
):
    """
    Wrapper per executar un experiment ARIMA/SARIMA, estàtic o rolling, i guardar resultats.

    Args:
        df (pd.DataFrame): Dades amb 'data' i 'valor'.
        params (dict): Diccionari amb els hiperparàmetres.
        save_root (str): Carpeta on guardar resultats.
    """


    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    # Silenciar warnings concrets
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


    # Comprovació bàsica
    if 'data_inici_pred' not in params:
        raise ValueError("⚠️ Falta el paràmetre obligatori 'data_inici_pred'")

    # Ruta de guardat
    nom_exp = construir_nom_experiment_arima(params, prefix=params.get('id', ''))
    save_path = os.path.join(save_root, nom_exp)

    # Evita repetir si ja està fet
    if os.path.exists(os.path.join(save_path, "metrics.csv")):
        print(f"❌ Ja existeix: {nom_exp}")
        return

    # print(f"🚀 Executant experiment: {nom_exp}") # (Ho comentem perq sino es mostra doble ja que tmb esta al launcher)

    # Rolling Forecast
    if params.get('rolling', False):
        pipeline_rolling_forecast_arima(
            df=df,
            data_inici_pred=params['data_inici_pred'],
            data_final_pred=params.get('data_final_pred'),
            n_passos=params.get('n_passos', 24),
            dies_entrenament=params.get('dies_entrenament', 60),
            p=params.get('p', 1),
            d=params.get('d', 1),
            q=params.get('q', 1),
            P=params.get('P'),
            D=params.get('D'),
            Q=params.get('Q'),
            s=params.get('s'),
            pas_pred=params.get('pas_pred', 1),
            plot_dies_ant=params.get('plot_dies_ant', 0),
            save_path=save_path,
            verbose=True,
            show=False
        )

    # Estàtic
    else:
        pipeline_arima(
            df=df,
            data_inici_pred=params['data_inici_pred'],
            data_final_pred=params.get('data_final_pred'),
            n_passos=params.get('n_passos', 24),
            dies_entrenament=params.get('dies_entrenament', 60),
            p=params.get('p', 1),
            d=params.get('d', 1),
            q=params.get('q', 1),
            P=params.get('P'),
            D=params.get('D'),
            Q=params.get('Q'),
            s=params.get('s'),
            plot_dies_ant=params.get('plot_dies_ant', 0),
            save_path=save_path,
            verbose=True,
            show=False
        )

    print(f"✅ Experiment completat: {nom_exp}")
