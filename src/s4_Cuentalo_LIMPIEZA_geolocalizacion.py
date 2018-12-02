# coding: utf-8


import pandas as pd
import numpy as np
import json

# ### Bajamos paises y ciudades
from country_city_sets import *

# #### Ahora el codigo para identificar pais/ciudad
from utils import *
import itertools

def clean_geolocate(pickle_file, locations_file, output_pickle):
    # # Getting some info from JSON file
    #df = pd.read_pickle("../pickles/cuentalo_clean_1.pkl")
    df = pd.read_pickle(pickle_file)

    # # Vamos a geolocalizar tweets

    final_locations_pd=pd.read_csv(locations_file) #"../intermediate_data/final_locations.csv")

    # paises unicos
    unique_countries=pd.concat([final_locations_pd[final_locations_pd['location_diagnosis']>0].pais_clean, final_locations_pd[final_locations_pd['parent_location_diagnosis']>0].parent_pais_clean]).dropna().unique()
    countries['clean_name_en']=countries.name_en.apply(clean_string)
    countries['clean_name']=countries.name.apply(clean_string)

    for country in unique_countries:
        subset=countries[(countries.clean_name==country) | (countries.clean_name_en==country)]
        if (len(subset==1)):
            country_dupes_dict[country]=subset.name.values[0]
        else:
            print ("error con ",country)
            print (subset)

    corrected_df = final_locations_pd.copy()
    def cleany(country):
        return country_dupes_dict[country] 
        
    corrected_df.pais_clean = final_locations_pd[(~final_locations_pd.pais_clean.isnull()) & (final_locations_pd.location_diagnosis>0)].pais_clean.apply(cleany)
    corrected_df.parent_pais_clean = final_locations_pd[(~final_locations_pd.parent_pais_clean.isnull()) & (final_locations_pd.parent_location_diagnosis>0)].parent_pais_clean.apply(cleany)

    # ciudades unicas
    unique_cities=pd.concat([final_locations_pd[final_locations_pd['location_diagnosis']>0].ciudad_clean, final_locations_pd[final_locations_pd['parent_location_diagnosis']>0].parent_ciudad_clean]).dropna().unique()

    cities['clean_name']=cities.name.apply(clean_string)

    # quitamos duplicados y nombres raros de ciudades
    for city in unique_cities:
        subset=cities[(cities.clean_name==city) ]
        if (len(subset==1)):
            city_dupes_dict[city]=subset.name.values[0]
        else:
            if city not in city_dupes_dict:
                print ("error con ",city,len(subset.name.values))

    city_dupes_dict['sinaloa']='Sinaloa'


    def cleany_city(city):
        return city_dupes_dict[city] 
        
    corrected_df.ciudad_clean = final_locations_pd[(~final_locations_pd.ciudad_clean.isnull()) & (final_locations_pd.location_diagnosis>0)].ciudad_clean.apply(cleany_city)
    corrected_df.parent_ciudad_clean = final_locations_pd[(~final_locations_pd.parent_ciudad_clean.isnull()) & (final_locations_pd.parent_location_diagnosis>0)].parent_ciudad_clean.apply(cleany_city)

    # grabamos just in case
    #corrected_df.to_csv("../intermediate_data/final_locations_clean.csv")

    # ahora adjudicamos
    full_df=pd.concat([df,corrected_df.set_index(['tweet_id'])],axis=1)
    full_df.to_pickle(output_pickle) #"../pickles/cuentalo_json_con_pais_ciudad_limpios.pkl")


    error_codes=[
    (-8, 'este error no deberia aparecer a menos que no hayamos visto todos los casos'),
    (-7, 'se mencionan varias ciudades y paises y no pudimos discriminar'),
    (-6, 'no hay este codigo'),
    (-5, 'se mencionan varias ciudades y no pais y no sabemos cual puede ser'),
    (-4, 'se menciona una ciudad y muchos paises y no podemos distinguir'),
    (-3, 'se menciona ciudad y no podemos discriminar entre varios posibles paises  '),
    (-2, 'no se menciona nada conocido'),
    (-1, 'sin info - texto vacio'),
    (1, 'sin info - pais sacado del time_zone'),
    (2, 'solo se menciona una region'),
    (3, 'solo se menciona un pais '),
    (4, 'se menciona una ciudad y una region'),
    (5, 'se menciona ciudad y de alli se deduce el pais'),
    (6, 'se menciona ciudad y el pais se deduce del tz'),
    (7, 'se menciona ciudad pero no sabemos pais'),
    (8, 'se menciona ciudad pero desempatamos pais con el tz'),
    (9, 'se menciona ciudad pero pueden ser varios paises -- se asume españa'),
    (10, 'se menciona ciudad pero pueden ser varios paises -- se asume UK'),
    (11, 'se menciona ciudad y el mismo pais varias veces'),
    (12, 'se menciona una ciudad y un pais'),
    (13, 'se menciona ciudad y desempatamos pais con tz'),
    (14, 'se menciona una ciudad y varios paises y desempatamos mirando a que pais corresponde la ciudad  '),
    (15, 'se menciona una ciudad y muchos paises -- si está se asume españa (1ero)'),
    (16, 'se menciona una ciudad y muchos paises -- si está se asume UK (2do)'),
    (17, 'se menciona una ciudad y muchos paises -- si está se asume mexico (3ero)'),
    (17.5, 'se menciona una ciudad y muchos paises -- si está se asume argentina (4to)'),
    (18, 'se mencionan varias ciudades y no pais y desempatamos con tz'),
    (19, 'se mencionan varias ciudades y no pais y desempatamos con el pais correspondiente que aparezca mas veces'),
    (19.5, 'se mencionan varias ciudades y no pais y elegimos la ciudad mas larga (caracteres)'),
    (20, 'se menciona un pais y varias ciudades y desempatamos viendo cual corresponde'),
    (21, 'se menciona una ciudad A y otra ciudad B que es pais, cogemos A,B '),
    (22, 'se mencionan varias ciudades y un pais y elegimos la ciudad mas larga (caracteres)'),
    ]
    for n in range(len(error_codes)):
        val, text = error_codes[n]
        if val>0:
            error_codes.append((val+0.001,text+", PAIS NO PEGA CON CIUDAD"))
            
    error_codes_df=pd.Series({k*1.000: v for k,v in error_codes})
    error_stats=pd.DataFrame()
    error_stats['error']=error_codes_df.sort_values()
    error_stats['cantidad']=full_df.groupby('location_diagnosis')['id'].count().sort_values()

    print ("ERRORES Y CLASIFICACION")
    print(error_stats.sort_values('cantidad').dropna())

