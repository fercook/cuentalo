import s1_Cuentalo_generacion_de_dataframe as PREPARE
import s2_basic_cleanup as CLEAN
import s3_Cuentalo_geolocalizacion_de_tweets as FIND_GEOLOCS
import s4_Cuentalo_LIMPIEZA_geolocalizacion as GEOLOCATE
import s5_integrate as INTEGRATE

# main dataset
#PREPARE.generate_dataframe("../data/Aniol-Maria-cuentalo-search-20180427_20180513.jsonl","../pickles/cuentalo_json_to.pkl")
#CLEAN.clean_pickle("../pickles/cuentalo_json_to.pkl", "../pickles/cuentalo_clean_1.pkl")
#FIND_GEOLOCS.localize_tweets("../pickles/cuentalo_clean_1.pkl","../intermediate_data/final_locations.csv")
#GEOLOCATE.clean_geolocate("../pickles/cuentalo_clean_1.pkl","../intermediate_data/final_locations.csv","../pickles/cuentalo_json_con_pais_ciudad_limpios.pkl")

# extension missing days
PREPARE.generate_dataframe("../data/misssing_tweets.jsonl","../pickles/missing_tweets.pkl")
CLEAN.clean_pickle("../pickles/missing_tweets.pkl", "../pickles/missing_tweets_clean.pkl")
FIND_GEOLOCS.localize_tweets("../pickles/missing_tweets_clean.pkl","../intermediate_data/missing_tweets_clean_locations.csv")
GEOLOCATE.clean_geolocate("../pickles/missing_tweets_clean.pkl","../intermediate_data/missing_tweets_clean_locations.csv","../pickles/missing_tweets_con_pais_ciudad_limpios.pkl")

INTEGRATE.integrate(["../pickles/cuentalo_json_con_pais_ciudad_limpios.pkl","../pickles/missing_tweets_con_pais_ciudad_limpios.pkl"],"../pickles/dataset_previo_final.pkl")
