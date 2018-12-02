import pandas as pd 

def integrate(df_pickle_list,output_df):
	dfs=[]
	for df in df_pickle_list:
		dfs.append(pd.read_pickle(df))
	df_full = pd.concat(dfs)
	df_full.to_pickle(output_df)
