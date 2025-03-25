### SCRIPT PER DESCARREGAR DADES DE LA WEB DE LA SCM I GUARDAR-LES EN UN FITXER ###

#Guarda totes les dades en un fitxer JSON

# Descarrega de les dades de temperatura

#################### LLIBRERIES ####################

import json
import os
import sys
import time
import datetime
import requests
import numpy as np
import pandas as pd


#################### CREAR DIRECTORIS ####################

# Estació que es baixarà les dades i dates que es volen baixar
# Paràmetres de la petició

data_inici = '1998-01-01' 
data_final = '2024-12-31'
est = 'Z1' # Bonaigua
var = 'T'
codi_var = 32


#################### CREAR DIRECTORIS ####################

    ######## Carpeta i fitxers dades
 
# Creem la carpeta on posarem les dades de cada estació i dia
carpeta_dades = f'Dades_{var}_estacions_xema_{est}'

# Ruta on esta el script actual
__path__ = os.path.dirname(os.path.realpath(__file__)) # Si es fitxer .py
# __path__ = os.getcwd() # Si es fitxer .ipynb
path_name = os.path.join(__path__, carpeta_dades)
data_path = os.path.join(path_name)

# Verifiquem si existeix la carpeta, si no existeix la creem
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)


    ######## Carpeta i fitxers metadades

# Creem la carpeta on posarem les metadades de l'estació i variable
meta_path_name = 'Metadades'
meta_data_path = os.path.join(data_path, meta_path_name) 

# Verifiquem si existeix la carpeta, si no existeix la creem
if not os.path.exists(meta_data_path):
    os.makedirs(meta_data_path, exist_ok=True)

# Nom del fitxer on guardarem les metadades
file_name_meta = f'SCM_metadades_{var}_{est}.json'

# Guardem les metadades al directori Dades Mensuals SCM
file_path_meta = os.path.join(meta_data_path, file_name_meta)

# URL de les metadades
url_meta = f'http://smcawrest01:8080/ApiRestInterna/xema/v1/mesurades/metadades/estacions/{est}/variables/{codi_var}?xarxa=1'


#################### DESCARREGA DE LES METADADES ####################

if not os.path.exists(file_path_meta):

    print('-----------------------------------')
    print(f'Descarregant metadades \n')

    # Fem la consulta
    response = requests.get(url_meta)

    # Si la resposta es correcte, guardem les dades en un fitxer
    if response.status_code == 200:
        
        print(f'METADATES Descarregades correctament \n')
        print('-----------------------------------')
        # Guardem les dades en un fitxer json
        with open(file_path_meta, 'w') as file:
            json.dump(response.json(), file, indent=4)

    else:
        error = json.loads(response.text)
        error_message = error['message']
        print(f' Error consultant metadades --> {error_message} \n')
        print('-----------------------------------')

else:
    print('-----------------------------------')
    print(f'Metadades ja descarregades \n')
    print('-----------------------------------')


#################### DESCARREGA DE LES DADES ####################

# Nom del fitxer on guardarem les dades
file_name = f'SCM_{var}_{est}.json'

# Ruta on guardarem el fitxer
file_path = os.path.join(data_path, file_name)

# Si el fitxer no existeix, el creem i descarreguem les dades
if not os.path.exists(file_path):
    print('-----------------------------------')
    print(f'Descarregant dades de estació {est} i variable {var} de la data {data_inici} a {data_final} \n')
    print('-----------------------------------')

    dades = []  # Llista per emmagatzemar totes les dades
    
    for date in pd.date_range(start=data_inici, end=data_final, freq='D'):
        di = date.strftime('%Y-%m-%dT00:00Z').replace(':','%3A')
        df = date.strftime('%Y-%m-%dT23:59Z').replace(':','%3A')

        url_data = f'http://smcawrest01:8080/ApiRestInterna/xema/v1/mesurades/dades/estacions/{est}/variables/{codi_var}?din={di}&dfi={df}&xarxa=1&nv=false'

        # Fem la consulta
        response = requests.get(url_data)
    
        if response.status_code == 200:
            dades.append(response.json())  # Afegim les dades a la llista
        else:
            error = json.loads(response.text)
            error_message = error['message']
            print(f' Error consultant dades {date.strftime("%Y-%m-%d")}: {error_message} \n')
            print('-----------------------------------')
    
    # Guardem totes les dades en un sol fitxer JSON
    with open(file_path, 'w') as file:
        json.dump(dades, file, indent=4)

    print('Descarrega de dades finalitzada \n')
    print('-----------------------------------')

else: # Si el fitxer ja existeix
    print('-----------------------------------')
    print(f'Dades ja descarregades i/o fitxer ja creat \n')
    print('-----------------------------------')
