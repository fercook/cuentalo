import pandas as pd
import numpy as np
import json
import datetime

def clean_pickle(infile,outfile):
	df = pd.read_pickle(infile)

	# limpieza datos basicos, llenamos NANs y procesamos fechas
	fechas = ['created_at','user_created_at', \
	'parent_created_at','parent_user_created_at']

	nans_a_int64 = ['id', 'user_id',  \
			'parent_id', 'parent_user_id']

	nans_a_int32 = ['user_followers_count','user_statuses_count' ,'retweet_count', 'favorite_count', 'quote_count', \
	'parent_user_followers_count','parent_user_statuses_count', 'parent_retweet_count', 'parent_favorite_count', 'parent_quote_count']

	nans_a_float = ['lat','lon','parent_lat','parent_lon']

	for col in fechas:
		print ("cleaning",col)
		df[col]=pd.to_datetime(df[col], infer_datetime_format=True)

	for col in nans_a_int64:
		print ("cleaning",col)
		df[col]=df[col].fillna(0).astype(np.int64)

	for col in nans_a_int32:
		print ("cleaning",col)
		df[col]=df[col].fillna(-1).astype(np.int32)

	for col in nans_a_float:
		print ("cleaning",col)
		df[col]=df[col].fillna(-999).astype(np.float32)

	# algunos tweets tienen actividad rara en el RT count, es porque son capturados mientras estan creciendo
	# sus numeros
	# ponemos el maximo pero podria ser util para saber cosas de dinamica

	'''df['orig_retweet_count']=df['retweet_count']

	retweets=df[df['tweet_type']=='retweet']
	grouped_retweets=retweets.groupby('parent_id')
	correct_count = grouped_retweets['retweet_count'].max()
	'''
	df.to_pickle(outfile)

"""
clean_pickle("../pickles/cuentalo_json_to.pkl", "../pickles/cuentalo_clean_1.pkl")
clean_pickle("../pickles/cuentalo_json_to_extra_1.pkl", "../pickles/cuentalo_clean_extra_1.pkl")
clean_pickle("../pickles/cuentalo_json_to_extra_2.pkl", "../pickles/cuentalo_clean_extra_2.pkl")
"""