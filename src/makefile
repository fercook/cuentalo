
all: ../pickles/cuentalo_json_con_pais_ciudad_limpios.pkl

generate: ../pickles/cuentalo_json_to.pkl #../pickles/cuentalo_json_to_extra_1.pkl ../pickles/cuentalo_json_to_extra_2.pkl

clean: ../pickles/cuentalo_clean_1.pkl #../pickles/cuentalo_clean_extra_1.pkl ../pickles/cuentalo_clean_extra_2.pkl

geolocalize: ../intermediate_data/final_locations.csv

clean_geoloc: ../pickles/cuentalo_json_con_pais_ciudad_limpios.pkl

../pickles/cuentalo_json_to.pkl: ../data/Aniol-Maria-cuentalo-search-20180427_20180513.jsonl
	python 01_Cuentalo_generacion_de_dataframe.py 

../pickles/cuentalo_clean_1.pkl: ../pickles/cuentalo_json_to.pkl
	python 02_basic_cleanup.py 

../intermediate_data/final_locations.csv: ../pickles/cuentalo_clean_1.pkl
	python 04_Cuentalo_geolocalizacion_de_tweets.py

../pickles/cuentalo_json_con_pais_ciudad_limpios.pkl: ../pickles/cuentalo_clean_1.pkl ../intermediate_data/final_locations.csv
	python 04B_Cuentalo_LIMPIEZA_geolocalizacion.py