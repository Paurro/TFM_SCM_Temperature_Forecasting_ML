
### SCRIPT PER DESCARREGAR DADES DE LA WEB DE LA SCM I GUARDAR-LES EN UN FITXER ###


# Descarrega de les dades de vent i pluja de les estacions de la SCM
# Descarrega de dades d'estacions del Delta de l'Ebre


#################### LLIBRERIES ####################

import json
import os
import sys
import time
import datetime
import requests
import numpy as np
import pandas as pd



#################### PARAMETRES I CODIS ####################

# URL Consulta dades mensuals 1 estacio en un mes determinat

# Exemple de URL
# url = 'https://api.meteo.cat/xema/v1/variables/estadistics/diaris/1000?codiEstacio=U9&any=2022&mes=07'

# URL base
url_base = 'https://api.meteo.cat/xema/v1/variables/estadistics/diaris/'


# Api key
key = 'mKcgBGwfKc6Zb0lwKGb763D9LKOFhodf8qsnhRIP'


# Creem la llista de dates a partir de la data inicial dels events meteorologics importants del Delta de l'Ebre 

events = [
    ["26/11/2016", "28/11/2016", "Altes Precipitacions, novembre 2016"],
    ["19/01/2017", "23/01/2017", "Temporal Bruno (alta precipitació y vent), Gener 2017"],
    ["18/10/2017", "19/10/2017", "Alta Precipitación, Octubre 2017"],
    ["02/01/2018", "04/01/2018", "Ventada, Gener 2018"],
    ["14/10/2018", "15/10/2018", "Ex Huracà Leslie (Ventada), Octubre 2018"],
    ["18/10/2018", "20/10/2018", "Aiguats a l'Ebre (Altes precipitacions), Octubre 2018"],
    ["14/11/2018", "16/11/2018", "Alta precipitació, Novembre 2018"],
    ["01/02/2019", "03/02/2019", "Ventada, Febrer 2019"],
    ["22/10/2019", "23/10/2019", "Altes precipitacions, octubre 2019"],
    ["03/12/2019", "05/12/2019", "Altes precipitacions i vent, desembre 2019"],
    ["19/01/2020", "21/01/2020", "Temporal Gloria (Altes precipitacions i vent, gener 2020)"],
    ["01/09/2020", "03/09/2020", "Aiguats Alcanar (Altes precipitacions), setembre 2020"],
    ["11/12/2020", "12/12/2020", "Tornado Horta de Sant Joan, (vent al delta), desembre 2020"],
    ["29/03/2020", "02/04/2020", "Altes precipitacions, març 2020"],
    ["09/01/2021", "12/01/2021", "Temporal Filomena (Altes precipitacions), gener 2021"],
    ["22/11/2021", "25/11/2021", "Altes precipitacions i vent (Tornado), novembre 2021"]
]


# Llista amb els events meteorologics importants del Delta de l'Ebre amb una extensio de dies inicial i final (5 abans i 5 despres)

ext_events = [
        ['21/11/2016', '03/12/2016', 'Altes Precipitacions, novembre 2016'],
        ['14/01/2017', '28/01/2017', 'Temporal Bruno (alta precipitació y vent), Gener 2017'],
        ['13/10/2017', '24/10/2017', 'Alta Precipitación, Octubre 2017'],
        ['28/12/2017', '09/01/2018', 'Ventada, Gener 2018'],
        ['09/10/2018', '20/10/2018', 'Ex Huracà Leslie (Ventada), Octubre 2018'],
        ['13/10/2018', '25/10/2018', "Aiguats a l'Ebre (Altes precipitacions), Octubre 2018"],
        ['09/11/2018', '21/11/2018', 'Alta precipitació, Novembre 2018'], 
        ['27/01/2019', '08/02/2019', 'Ventada, Febrer 2019'], 
        ['17/10/2019', '28/10/2019', 'Altes precipitacions, octubre 2019'], 
        ['28/11/2019', '10/12/2019', 'Altes precipitacions i vent, desembre 2019'], 
        ['14/01/2020', '26/01/2020', 'Temporal Gloria (Altes precipitacions i vent, gener 2020)'], 
        ['27/08/2020', '08/09/2020', 'Aiguats Alcanar (Altes precipitacions), setembre 2020'], 
        ['06/12/2020', '17/12/2020', 'Tornado Horta de Sant Joan, (vent al delta), desembre 2020'], 
        ['24/03/2020', '07/04/2020', 'Altes precipitacions, març 2020'], 
        ['04/01/2021', '17/01/2021', 'Temporal Filomena (Altes precipitacions), gener 2021'], 
        ['17/11/2021', '30/11/2021', 'Altes precipitacions i vent (Tornado), novembre 2021']
]
     

# Llista amb els codis de les estacions meteorologiques d'interpes i el seu nom
estacions = {
            'U9': 'L\'Aldea',
            'DL': 'Sant Jaume d\'Enveja',
            'UU': 'Amposta',
            'UW': 'La Ràpita',
            'UA': 'L\'Ametlla de Mar',
            'U7': 'Aldover',
            'X5': 'Roquetes',
            'C9': 'Mas de Barberans',
            'US': 'Alcanar',
            'UT': 'les Cases d\'Alcanar',
            'UX': 'Ulldecona - els Valentins'                                                                                                                                                                                                                       
            }

# Llista amb els codis de les variables d'interes i el seu nom
codis_variables_2 = {
                1300 : 'Precipitació acumulada diària (mm)', 
                1503 : 'Velocitat mitjana diària del vent 10 m (m/s)', 
                1505 : 'Velocitat mitjana diària del vent 2 m (m/s)',
                1509 : 'Direcció mitjana diària del vent 10 m (graus)', 
                1511 : 'Direcció mitjana diària del vent 2 m (s/m)', 
                1512 : 'Ratxa màxima diària del vent 10 m (m/s)', 
                1514 : 'Ratxa màxima diària del vent 2 m (m/s)', 
                1515 : 'Direcció de la ratxa màxima diària del vent 10 m (graus)', 
                1517 : 'Direcció de la ratxa màxima diària del vent 2 m (graus)'
                }




#################### CREAR DIRECTORIS ####################

# Creem la carpeta on posarem subcarpetes amb els fitxers json de cada event

__path__ = os.path.dirname(os.path.abspath(__file__))
path_name = os.path.join(__path__, 'Dades SCM Temporals Adversos Delta Ebre')
data_path = os.path.join(path_name)

# Verifiquem si existeix la carpeta, si no existeix la creem

if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)


# Crea un fitxer txt per posar el nom dels fitxers json que es crearan per obrir-los despres
# Guardarem aquest fitxer a la carpeta Dades Mensuals SCM

# noms_fitxers = 'noms_fitxers.txt'
# noms_fitxers_path = os.path.join(data_path, noms_fitxers)

# if not os.path.exists(noms_fitxers_path):
#     with open(noms_fitxers_path, 'w') as file:
#        pass

#################### DESCARREGAR METADADES ####################

# URL base metadades
url_meta = 'https://api.meteo.cat/xema/v1/variables/estadistics/diaris/metadades'

meta_path_name = os.path.join(__path__, 'Metadades i codis')
meta_data_path = os.path.join(meta_path_name)

# Verifiquem si existeix la carpeta, si no existeix la creem

if not os.path.exists(meta_data_path):
    os.makedirs(meta_data_path, exist_ok=True)

# Nom del fitxer on guardarem les metadades
file_name_meta = 'SCM_metadades_diaries_mensuals.json'


# Guardem les metadades al directori Dades Mensuals SCM
file_path_meta = os.path.join(meta_data_path, file_name_meta)

if not os.path.exists(file_path_meta):

    print('-----------------------------------')
    print(f'Descarregant metadades \n')

    # Fem la consulta

    response = requests.get(url_meta, headers = {'content-type':'application/json', 'X-Api-Key': key})

    # Si la resposta es correcte, guardem les dades en un fitxer

    if response.status_code == 200:

        # # # Afegir el nom del fitxer al fitxer noms_fitxers.txt

        # with open(noms_fitxers_path, 'a') as file:
        #     file.write(file_name_meta + '\n')
        #     file.close()
            
        print(f'METADATES Descarregades correctament \n')
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



#################### DESCARREGAR DADES ####################

        ### Extreu els mesos i anys dels events ###

# Recorrem la llista de events i per cada event descarreguem les dades de les estacions

for event in ext_events:
    event = event[2]

    # Extreure els mesos i anys del event a partir de la llista ext_events
    mesos = []
    anys = []

    for ext_event in ext_events:
            
        if ext_event[2] == event:
            dates = ext_event[:2]
            for date in dates:
                mesos.append(date.split('/')[1])
                anys.append(date.split('/')[2])
                
            if mesos[0] == mesos[1] and anys[0] == anys[1]:
                del mesos[1]
                del anys[1]
            break
       
    ############# Consulta dades de les estacions ##################
    
    #Creem la consulta per cada estacio i cada variable en fitxer json diferents

    # Creem el directori on guardarem els fitxers json
    # Directori creat a la mateixa carpeta on es troba aquest script

    # Creem una subcarpeta dins de la carpeta Dades Mensuals SCM per cada event
    # El nom de la subcarpeta sera el nom de l'event
        
    path_name = os.path.join(data_path, event)

    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)


    print('\n')
    print(f'Inici descarrega dades Servei Catala de Meteorologia \n')
    print(f'{event}')
    print(f'Mesos: {mesos}')
    print(f'Anys: {anys} \n')

    for estacio in estacions.keys():

        print('-----------------------------------')
        print(f'Estació: {estacions[estacio]} ({estacio}) : Inici descarrega dades \n')
            
        for codi in codis_variables_2.keys():

            for mes, any in zip(mesos, anys):

                # Creem un fitxer json per emmagatzemar les dades de l'estació en un únic fitxer

                file_name = 'SCM_' + estacio + '_' + str(codi) + '_' + any + '_' + mes + '.json'
                    
                # Crear fitxer a la carpeta Dades Mensuals SCM

                file_path = os.path.join(path_name, file_name)

                if not os.path.exists(file_path):

                    print(f'Descarregant dades {codis_variables_2[codi]}(codi: {codi})')
                        

                    # Creem la URL
                    url = url_base + str(codi) + '?codiEstacio=' + estacio + '&any=' + any + '&mes=' + mes
                    
                    # Fem la consulta
                    response = requests.get(url, headers = {'content-type':'application/json', 'X-Api-Key': key})

                    # Si la resposta es correcte, guardem les dades en un fitxer

                    if response.status_code == 200:
                                    
                        # Guardem les dades en un fitxer json
                        with open(file_path, 'w') as file:
                            json.dump(response.json(), file, indent=4)
                                
                        print(f'Descarregades correctament \n')


                    else:

                        error = json.loads(response.text)
                        error_message = error['message']
                        # print(f'Estació {estacio}: Error Consulta dades {codi} \n error : {error_message}')
                        print(f'Error Consulta dades --> {error_message} \n')

                else:
                    print(f'Dades {codis_variables_2[codi]}(codi: {codi}) ja descarregades \n')

        print(f'Estació: {estacions[estacio]} ({estacio}) : Fi descarrega dades \n')
        print('-----------------------------------')


print ('Fi descarrega dades SCM')
print('\n')


