import pandas as pd
import numpy as np
import json
import math
from datetime import timedelta

THRESHOLD_TROLLS =.75 #arriba de esto son tweets raros
ANGLE_NOISE = 1 #in degrees
INTERNAL_RADIUS = 0.7
BLUR_TESTIMONIOS = 0.2
THRESHOLD_BLUR_TESTIMONIO = 0.95
BLUR_APOYO = 0.2
THRESHOLD_BLUR_APOYO = 0.95
SCALE_APOYO = 0.7 
SCALE_OTROS = 2
PUSH_OTROS = 2
COLUMNAS_A_CONSERVAR = ['id','created_at','pais_clean','user_followers_count','retweet_count','full_text']

#
# generamos un dataset resumido
#
#
df_full=pd.read_pickle("../pickles/dataset_previo_final.pkl")

summary = df_full[df_full.tweet_type!='retweet'][COLUMNAS_A_CONSERVAR].copy()
# cambiar enters por puntos
summary.full_text=summary.full_text.apply(lambda x: x.replace('\n', '.') if x!=None else None)

#
# Probabilidades de categorias
probabilities = pd.read_csv("../data/df_predicted_probs_originals.csv",)

probabilities=probabilities[['id','pred_1a2a_persona','pred_apoyo','pred_otros_trolls','pred_fisico','pred_no_fisico','pred_otros']]
probabilities=probabilities[~probabilities.id.isna()]
probabilities.id= probabilities.id.astype(np.int)

# preparamos para el join
summary=summary.set_index('id')
probabilities = probabilities.set_index('id')
prob_summary = summary.join(probabilities)

### Corregimos la hora a hora local (donde se puede)

# from here https://en.wikipedia.org/wiki/List_of_time_zones_by_country
avg_tz_delta = {'España': 1,
  'Argentina': -3,
  'Colombia': -5,
  'México': -6,
  'Chile': -4,
  'Estados Unidos': -6.5,
  'Perú': -5,
  'Venezuela': -4,
  'Uruguay': -3,
  'Costa Rica': -6,
  'Ecuador': -5,
  'Puerto Rico': -4,
  'Reino Unido': 0,
  'Panamá': -5,
  'Brasil': -3,
  'Guatemala': -6,
  'Paraguay': -4,
  'Francia': 1,
  'El Salvador': -6,
  'Filipinas': 8,
  'Alemania': 1,
  'Italia': 1,
  'Cuba': -5,
  'Irlanda': 0,
  'Países Bajos': 1,
  'Honduras': -6,
  'Australia': 9,
  'Canadá': -6,
  'Nicaragua': -6,
  'Japón': 9,
  'Portugal': 0,
  'Rusia': 3,
  'Suiza': 1,
  'San Martín': -4,
  'Bolivia': -4,
  'Nigeria': -6,
  'India': 5.5,
  'Bélgica': 1,
  'Turquía': 3,
  'República Dominicana': -4,
  'Noruega': 1,
  'Grecia': 1,
  'Nueva Zelanda': 12,
  'Marruecos': 1,
  'Polonia': 1,
  'Suecia': 1,
  'Finlandia': 1,
  'Jamaica': -5,
  'Jersey': 0,
  'Corea del Sur': 9,
  'Sudáfrica': 2,
  'Andorra': 1,
  'Islandia': 0,
  'Rumania': 1,
  'Bangladés': 6,
  'Austria': 1,
  'Santa Lucía': -4,
  'Hong Kong': 8,
  'Indonesia': 7,
  'Egipto': 1,
  'Maldivas': 5,
  'Luxemburgo': 1,
  'Arabia Saudita': 3,
  'Palestina': 3
    }
def hora_local(row):
    hora_utc = row.created_at
    pais = row.pais_clean
    if pais in avg_tz_delta:
        delta = timedelta(0,3600*avg_tz_delta[pais])
        return hora_utc + delta
    else:
        return hora_utc
prob_summary['hora_local'] = prob_summary.apply(hora_local,axis=1)



#
#
#### Coordenadas para la grafica
#
#
####### X para grafica lineal
mindate=prob_summary.hora_local.min()
maxdate=prob_summary.hora_local.max()
prob_summary['lineal_date']=prob_summary.hora_local.apply(lambda x: (x-mindate).total_seconds())

prob_summary = prob_summary.sort_values(by='hora_local',ascending=True)
delta=60 #seconds, so I have 1440 groups per day
prob_summary['groups_linear']=(prob_summary['lineal_date']/delta).astype(int)
prob_summary['groups_circle']=prob_summary['groups_linear']%1440


#
# FILTRAMOS LOS TWEETS RAROS
prob_summary=prob_summary[prob_summary['pred_otros_trolls']<0.75]

#angle 
prob_summary['angle']=math.pi/2-2*math.pi*prob_summary.groups_circle/1440+ANGLE_NOISE*np.random.randn(len(prob_summary))/360

#radius
testimonio = BLUR_TESTIMONIOS*np.random.randn(len(prob_summary)) *( np.where(prob_summary['pred_1a2a_persona']>THRESHOLD_BLUR_TESTIMONIO,prob_summary['pred_1a2a_persona'],0))
# th: 0.5 y blur=0.3
# BLUR_TESTIMONIOS*np.random.randn(len(prob_summary)) *( (1+THRESHOLD_BLUR_TESTIMONIO)-prob_summary['pred_1a2a_persona'] )+ 
apoyo = SCALE_APOYO*(prob_summary['pred_apoyo']+BLUR_APOYO*np.random.randn(len(prob_summary))*( np.where(prob_summary['pred_apoyo']>THRESHOLD_BLUR_APOYO,prob_summary['pred_apoyo'],0)))
otros = SCALE_OTROS*np.exp(PUSH_OTROS*prob_summary['pred_otros_trolls'])
prob_summary['radius'] = testimonio +  apoyo +  otros  - INTERNAL_RADIUS

prob_summary['x'] = prob_summary.radius*np.cos(prob_summary.angle)
prob_summary['y'] = prob_summary.radius*np.sin(prob_summary.angle)


#
#
#
### grabamos 
#
cols = ['hora_local','pais_clean','user_followers_count','retweet_count','full_text','pred_1a2a_persona','pred_apoyo','pred_otros_trolls','pred_fisico','pred_no_fisico','pred_otros','x','y']
# reducimos un poco los numeros
floatcols = ['pred_1a2a_persona','pred_apoyo','pred_otros_trolls','pred_fisico','pred_no_fisico','pred_otros','x','y']
intcols = ['user_followers_count','retweet_count']
for f in floatcols:
    prob_summary[f] = prob_summary[f].round(5)
for i in intcols:
    prob_summary[i] = prob_summary[i].astype(np.int).astype('str')

# separamos en coordenadas e info extra por tweet
cols_for_coords = ['retweet_count','x','y']
other_cols = ['hora_local','pais_clean','user_followers_count','full_text','pred_1a2a_persona','pred_apoyo','pred_otros_trolls','pred_fisico','pred_no_fisico','pred_otros']

prob_summary[cols_for_coords].to_csv("OUT_coordenadas_cuentalo.csv")
prob_summary[other_cols].to_csv("OUT_details_cuentalo.csv")



"""
# esto se puede descomentar para ver graficas en Python
%matplotlib inline
import matplotlib as plt
pd.set_option("display.max_columns",999)
"""



""" # grafica de cebolla

def abc_to_rgb(A=0.0,B=0.0,C=0.0):
    ''' Map values A, B, C (all in domain [0,1]) to
    suitable red, green, blue values.'''
    return (min(B+C,1.0),min(A+C,1.0),min(A+B,1.0))
    
    
colores = [ abc_to_rgb(a,b,c)   for a,b,c in zip(prob_summary.pred_fisico, prob_summary.pred_no_fisico, prob_summary.pred_otros)]

plt.pyplot.figure(figsize=(20,20))
plt.pyplot.scatter(x=prob_summary.radius*np.cos(prob_summary.angle),c=colores,
                   y=prob_summary.radius*np.sin(prob_summary.angle),
                   s=2,alpha = 0.3)
plt.pyplot.xlim((-3,3))
plt.pyplot.ylim((-3,3))
plt.pyplot.show()



"""

""" #graficas de las probabilidades
import ternary
points = [(x[1],x[2],x[3]) for x in probabilities.to_records()]
figure, tax = ternary.figure()
figure.set_size_inches(8, 8)
tax.boundary(linewidth=0.01)
#tax.gridlines(multiple=5, color="blue")
tax.scatter(points, color='grey',s=0.2)
tax.scatter([(1,0,0)], marker='D', color='red', label="Testimonio")
tax.scatter([(0,1,0)], marker='D', color='blue', label="Apoyo")
tax.scatter([(0,0,1)], marker='D', color='green', label="Otros")
tax.legend()


points = [(x[4],x[5],x[6]) for x in probabilities.to_records()]
figure, tax = ternary.figure()
figure.set_size_inches(8, 8)
tax.boundary(linewidth=0.01)
#tax.gridlines(multiple=5, color="blue")
tax.scatter(points, color='grey',s=0.2)
tax.scatter([(1,0,0)], marker='D', color='red', label="Fisico")
tax.scatter([(0,1,0)], marker='D', color='blue', label="Psicologico")
tax.scatter([(0,0,1)], marker='D', color='green', label="Emocion")
tax.legend()


"""
