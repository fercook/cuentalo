
# coding: utf-8

import pandas as pd

### Bajamos paises y ciudades

# get unlocode from this: https://datahub.io/core/un-locode
un_country_codes=pd.read_csv("../external/unlocode/country-codes.csv")
un_city_codes=pd.read_csv("../external/unlocode/code-list.csv")
un_region_codes=pd.read_csv("../external/unlocode/subdivision-codes.csv")

# World countries: https://github.com/stefangabos/world_countries
countries_es=pd.read_csv("../external/countries_es.csv")
countries_en=pd.read_csv("../external/countries_en.csv")
countries_en.columns=['id', 'name_en', 'alpha2', 'alpha3']

countries=pd.merge(countries_en[['id', 'name_en']],countries_es[['id', 'name']],on="id",how="inner")

# Cities: http://www.unece.org/cefact/codesfortrade/codes_index.html, http://download.geonames.org/export/dump/
cities=pd.read_csv("../external/world-cities_csv.csv")

# limpiamos y estandarizamos acentos y tildes porque si no es un lio
# voy a copiar la limpieza de UNLOCODE
# 
diacritics_replacements = [
    [["À", "Á", "Â", "Ã", "Ä", "Å", "Æ"] ,"A"],
    [["Ç"], "C"],
    [["È", "É", "Ê", "Ë"], "E"],
    [["Ì", "Í", "Î", "Ï"] ,"I"],
    [["Ñ"], "N"],
    [["Ò", "Ó", "Ô", "Õ", "Ö", "Ø"], "O"],
    [["Ù", "Ú", "Û", "Ü"], "U"],
    [["Ý"], "Y"],
    [["à", "á", "â", "ã", "ä", "å", "æ"], "a"],
    [["ç"] ,"c"],
    [["è", "é", "ê", "ë"], "e"],
    [["ì", "í", "î", "ï"], "i"],
    [["ñ"], "n"],
    [["ò", "ó", "ô", "õ", "ö", "ø"], "o"],
    [["ù", "ú", "û", "ü"], "u"],
    [["ý", "ÿ"], "y" ],
    [[",","-","+",".","/","♥"], " "]
]

def clean_string(string):
    clean_str=string.lower()
    for originals, replacement in diacritics_replacements:
        for original in originals:
            clean_str=clean_str.replace(original,replacement)
    return clean_str

#
###
#
#vamos a intentar usar codigos para juntar paises en castellano e ingles
# 
code_of_country={}
country_from_code={}
for idx,row in countries.iterrows():
    code_of_country[clean_string(row['name_en'])]=row['id']
    code_of_country[clean_string(row['name'])]=row['id']
    country_from_code[row['id']]=clean_string(row['name_en'])

country_set=set(code_of_country.keys())
cities_set=set([clean_string(x) for x in cities.name.values])

# las dbs no son compatibles, arreglamos las diferencias
# correr esto en interactivo
'''errors=set()
for row,city in cities.iterrows():
    clean_city=clean_string(city['name'])
    clean_country=clean_string(city.country)
    if clean_country in code_of_country:
        pass
    else:
        errors.add(clean_country)
print (sorted(list(errors)))

a=list(code_of_country.keys())
a.sort()
[(x, code_of_country[x]) for x in a]'''

maxid=countries.id.max()

code_of_country['aland islands']=248
code_of_country['bonaire  saint eustatius and saba']=535
code_of_country['british virgin islands']=850
code_of_country['cape verde']=132
code_of_country['cocos islands']=166
code_of_country['curacao']=531
code_of_country['czech republic']=203
code_of_country['democratic republic of the congo']=180
code_of_country['east timor']=626
code_of_country['falkland islands']=238
code_of_country['ivory coast']=384
code_of_country['czech republic']=203
code_of_country['kosovo']=maxid+1
country_from_code[maxid+1]='kosovo'
maxid=maxid+1
code_of_country['moldova']=498
code_of_country['north korea']=408
code_of_country['palestinian territory']=275
code_of_country['republic of the congo']=180
code_of_country['russia']=643
code_of_country['saint helena']=654
code_of_country['saint martin']=663
code_of_country['south korea']=410
code_of_country['swaziland']=748
code_of_country['syria']=760
code_of_country['taiwan']=158
code_of_country['u s  virgin islands']=850
code_of_country['united kingdom']=826
code_of_country['united states']=840
code_of_country['vatican']=336

code_of_country['espanya']=724


# creamos un diccionario para saber a que paises pertenece una ciudad
country_of_city={}
for row,city in cities.iterrows():
    clean_city=clean_string(city['name'])
    clean_country=clean_string(city.country)
    if clean_country in code_of_country:
        clean_country_code=code_of_country[clean_country]
    if clean_city in country_of_city:
        country_of_city[clean_city]=country_of_city[clean_city]+[clean_country_code]
    else:
        country_of_city[clean_city]=[clean_country_code]

def check_city_errors():
    errors=set()
    for row,city in cities.iterrows():
        if clean_country not in code_of_country:
            errors.add(clean_country)
    print (errors)

#### Arreglamos errores puntuales encontrados a mano

# agregamos ciudades
ciudades_a_agregar = [ 
    ['san sebastian','spain'],
    ['seville','spain'],
    ['los navalmorales','spain'],
    ['las palmas','spain'],
    ['cabrales','spain'],
    ['serranillos del valle','spain'],
    ['godelleta','spain'],
    ['xixon','spain'],
    ['almazora','spain'],
    ['mollet del valles','spain'],
    ['villaturiel','spain'],
    ['mollerussa','spain'],
    ['murchante','spain'],
    ['madriz','spain'],
    ['madriles','spain'],
    ['bcn','spain'],
    ['torrellas','spain'],
    ['alacant','spain'],
    ['huescar','spain'],
    ['vallekas','spain'],
    ['compostela','spain'],
    ['villarreal','spain'],
    ['nava de la asuncion','spain'],
    ['viana do bolo','spain'],
    ['Santiago Bernabéu','spain'],
    ['tucuman','argentina'],
    ['catamarca','argentina'],
    ['cba','argentina'],
    ['buenos aire','argentina'],
    ['bs as','argentina'],
    ['boedo','argentina'],
    ['berazategui','argentina'],
    ['burzaco','argentina'],
    ['caballito','argentina'],
    ['mataderos','argentina'],
    ['merlo','argentina'],
    ['trenque Lauquen','argentina'],
    ['lanus','argentina'],
    ['villa urquiza','argentina'],
    ['bariloche','argentina'],
    ['lomas de zamora','argentina'],
    ['la plata','argentina'],
    ['santiago de cali','colombia'],
    ['cali','colombia'],
    ['new york','united states'],
    ['nueva york','united states'],
    ['viena','austria'],
    ['distrito federal','mexico'],
    ['metropolitana de santiago','chile'],
    ['londres','united kingdom'],
    ['sonora','mexico'],
    ['sinaloa', 'mexico'],
    ['manchester', 'united kingdom'],
    ['washington', 'united states'],
    ['san martin', 'argentina']
]

for city,country in ciudades_a_agregar:
    cities_set.add(city)
    country_of_city[city]=[code_of_country[country]]
    
if code_of_country['argentina'] not in country_of_city['santa fe']: 
    country_of_city['santa fe'].append(code_of_country['argentina'])

#agregamos regiones
regiones={}
regiones['chubut']=code_of_country['argentina']
regiones['arg']=code_of_country['argentina']
regiones['🇦🇷']=code_of_country['argentina']
regiones['arg🇦🇷']=code_of_country['argentina']
regiones['🇲🇽']=code_of_country['mexico']
regiones['🇵🇷']=code_of_country['cuba']
regiones['🇨🇱']=code_of_country['chile']

regiones['asturias']=code_of_country['spain']
regiones['catalunya']=code_of_country['spain']
regiones['euskal']=code_of_country['spain']
regiones['donostia']=code_of_country['spain']
regiones['catalans']=code_of_country['spain']
regiones['catalans']=code_of_country['spain']
regiones['tenerife']=code_of_country['spain']
regiones['andalucia']=code_of_country['spain']
regiones['galicia']=code_of_country['spain']
regiones['galiza']=code_of_country['spain']
regiones['lar']=code_of_country['spain']
regiones['asturies']=code_of_country['spain']
regiones['canarias']=code_of_country['spain']
regiones['cantabria']=code_of_country['spain']
regiones['navarra']=code_of_country['spain']
regiones['islas canarias']=code_of_country['spain']
regiones['pucela']=code_of_country['spain']
regiones['alicante']=code_of_country['spain']
regiones['republica espanola']=code_of_country['spain']
regiones['mallorca']=code_of_country['spain']
regiones['republica catalana']=code_of_country['spain']
regiones['catalonia']=code_of_country['spain']
regiones['malasana']=code_of_country['spain']
regiones['burgos']=code_of_country['spain']
regiones['espanya']=code_of_country['spain']
regiones['pais vasco']=code_of_country['spain']
regiones['bizkaia']=code_of_country['spain']
regiones['gran canaria']=code_of_country['spain']
regiones['sant esteve de les roures']=code_of_country['spain']
regiones['extremadura']=code_of_country['spain']
regiones['grana']=code_of_country['spain']
regiones['euskadi']=code_of_country['spain']

regiones['uruwhy']=code_of_country['uruguay']   

regiones['holanda']=code_of_country['netherlands']
regiones['inglaterra']=code_of_country['united kingdom']
regiones['england']=code_of_country['united kingdom']
regiones['california']=code_of_country['united states']
regiones['usa']=code_of_country['united states']
regiones['united states']=code_of_country['united states']
regiones['florida']=code_of_country['united states']
regiones['ca']=code_of_country['united states']
regiones['mx']=code_of_country['mexico']

#reemplazamos casos raros
country_of_city['granada']=[code_of_country['spain']]
country_of_city['madrid']=[code_of_country['spain']]
country_of_city['lima']=[code_of_country['peru']]
country_of_city['los angeles']=[code_of_country['united states']]
country_of_city['london']=[code_of_country['united kingdom']]
country_of_city['sydney']=[code_of_country['australia']]
country_of_city['entre rios']=[code_of_country['argentina']]

# quitamos ciudades que significan cosas o dan ruido
ciudades_a_remover = ['san','un','una','of','fes','sur','tanga','bar','colombia','andalucia','venezuela','puerto rico','lar']

for city in ciudades_a_remover:
    country_of_city.pop(city,None)
    if city in cities_set: cities_set.remove(city)


# Vamos a marcar algunos husos horarios como los mas probables para algun pais en particular
country_from_time_zone={ x: None for x in [
    'Caracas', 'Pacific Time (US & Canada)', 'Amsterdam',
       'Hawaii', 'Athens', 'Greenland', 'Brasilia', 'Madrid',
       'Buenos Aires', 'Bogota', 'Dublin', 'Central Time (US & Canada)',
       'Belgrade', 'Central America', 'Ljubljana', 'Paris', 'Mexico City',
       'London', 'Stockholm', 'America/Bogota', 'Quito',
       'America/Argentina/Buenos_Aires', 'Bern', 'Santiago',
       'Atlantic Time (Canada)', 'Arizona', 'Casablanca',
       'Eastern Time (US & Canada)', 'Europe/Madrid',
       'Mountain Time (US & Canada)', 'America/Montevideo', 'Lisbon',
       'America/Guatemala', 'Tijuana', 'Wellington', 'Mid-Atlantic',
       'Alaska', 'America/Mexico_City', 'Berlin', 'West Central Africa',
       'Lima', 'Brisbane', 'Monterrey', 'America/Guayaquil',
       'International Date Line West', 'Brussels', 'Azores',
       'America/Hermosillo', 'Midway Island', 'America/Panama', 'Zagreb',
       'New Delhi', 'La Paz', 'Europe/Amsterdam', 'Copenhagen', 'Rome',
       'CET', 'Newfoundland', 'Vienna', 'America/Santiago', 'Georgetown',
       'America/Lima', 'Hong Kong', 'America/Manaus', 'Cape Verde Is.',
       'Europe/London', 'UTC', 'Almaty', 'Minsk', 'Budapest',
       'Guadalajara', 'America/Caracas', 'Mazatlan', 'Monrovia',
       'Baghdad', 'Moscow', 'Nairobi', 'Cairo', 'America/Barbados',
       'Skopje', 'Bratislava', 'Osaka', 'Warsaw', 'Krasnoyarsk', 'Prague',
       'Sarajevo', "Nuku'alofa", 'Tehran', 'Africa/Ceuta', 'Bangkok',
       'New Caledonia', 'Solomon Is.', 'America/Asuncion', 'Bucharest',
       'Samoa', 'Fiji', 'America/La_Paz', 'Sydney', 'Edinburgh', 'Kuwait',
       'Kamchatka', 'Irkutsk', 'America/New_York', 'Abu Dhabi',
       'Chihuahua', 'Kabul', 'Kathmandu', 'Saskatchewan', 'Sofia',
       'Canberra', 'Harare', 'Volgograd', 'Muscat',
       'America/Buenos_Aires', 'Europe/Berlin', 'Kyiv', 'Singapore',
       'Melbourne', 'Europe/San_Marino', 'Helsinki', 'Auckland', 'Seoul',
       'America/Sao_Paulo', 'Europe/Luxembourg', 'Atlantic/Canary',
       'Yerevan', 'Islamabad', 'Tokyo', 'GMT+2', 'Beijing', 'Vladivostok',
       'Istanbul', 'Dhaka', 'America/Cordoba', 'Europe/Athens', 'Darwin',
       'America/Araguaina', 'Marshall Is.', 'Novosibirsk',
       'Europe/Tirane', 'Indiana (East)', 'Jakarta', 'Europe/Paris',
       'Jerusalem', 'Europe/Rome', 'Riyadh', 'Chennai', 'Europe/Brussels',
       'Europe/Stockholm', 'America/Detroit', 'America/Chihuahua',
       'Rangoon', 'Magadan', 'Vilnius', 'GMT-5', 'Astana',
       'Europe/Belgrade', 'Adelaide', 'America/Havana', 'Baku',
       'Sri Jayawardenepura', 'Tallinn', 'America/El_Salvador', 'Karachi',
       'America/Los_Angeles', 'Tbilisi', 'Sapporo', 'Hanoi', 'Mumbai',
       'Europe/Bratislava', 'WET', 'GMT', 'Yakutsk', 'Ekaterinburg',
       'America/Puerto_Rico', 'Urumqi', 'Pretoria',
       'America/Santo_Domingo', 'ART', 'America/Costa_Rica', 'Tashkent',
       'AST', 'America/Cancun', 'Pacific/Auckland', 'America/Chicago',
       'Chongqing', 'America/Managua', 'America/Noronha',
       'Africa/Windhoek', 'GMT-3', 'Europe/Andorra', 'Hobart', 'Perth',
       'Australia/Adelaide', 'Kuala Lumpur', 'America/Denver', 'GMT-6',
       'America/Tijuana', 'America/Phoenix', 'America/Monterrey',
       'Pacific/Midway', 'CDT', 'ECT', 'America/Anguilla', 'EST',
       'Asia/Taipei', 'America/Aruba', 'Ulaan Bataar', 'GMT-4',
       'Port Moresby', 'Europe/Copenhagen', 'Guam', 'Europe/Sarajevo',
       'Asia/Tokyo', 'Asia/Shanghai', 'America/Tegucigalpa',
       'Atlantic/Azores', 'America/Glace_Bay', 'America/Belem',
       'America/Halifax', 'JST', 'Taipei', 'Atlantic/South_Georgia',
       'Atlantic/Stanley', 'Asia/Dubai', 'Riga', 'Asia/Kolkata',
       'America/Bahia_Banderas', 'America/Godthab', 'CST',
       'America/Toronto', 'PDT', 'America/Mazatlan', 'Africa/Casablanca',
       'Asia/Amman', 'Kolkata', 'America/Boise', 'Asia/Seoul',
       'Africa/Algiers', 'Africa/Nairobi', 'Europe/Moscow', 'GMT+1',
       'Asia/Calcutta', 'Kiev', 'Asia/Ho_Chi_Minh']
}
specific_tz={
    'Caracas': 'venezuela', 
    'Madrid': 'espana',
    'Buenos Aires': 'argentina', 
    'Bogota': 'colombia',
    'Mexico City': 'mexico',
    'America/Bogota': 'colombia', 
    'Quito': 'ecuador',
    'America/Argentina/Buenos_Aires': 'argentina',
    'Santiago': 'chile',
    'Europe/Madrid': 'espana',
    'America/Montevideo': 'uruguay',
    'America/Mexico_City': 'mexico',
    'Lima': 'peru', 
    'Monterrey': 'mexico',
    'La Paz': 'bolivia',
    'America/Santiago': 'chile',
    'America/Lima': 'peru',
    'Guadalajara': 'mexico', 
    'America/Caracas': 'venezuela'
}

for tz in specific_tz:
    country_from_time_zone[tz]=code_of_country[specific_tz[tz]]



########## 
#
# duplicados y nombres raros de ciudades y paises

city_dupes_dict={
    'villarreal': 'Vila-real',
    'bcn': 'Barcelona',
    'metropolitana de santiago': 'Santiago',
    'bs as': 'Buenos Aires',
    'nava de la asuncion': 'Nava de la Asunción',
    'buenos aire': 'Buenos Aires',
    'las palmas': 'Las Palmas de Gran Canaria',
    'new york': 'New York City',
    'tordesillas': 'Tordesillas',
    'tucuman': 'San Miguel de Tucumán',
    'san sebastian': 'Donostia / San Sebastián',
    'los navalmorales': 'Los Navalmorales',
    'alacant': 'Alicante',
    'xixon': 'Gijón',
    'villaturiel': 'Villaturiel',
    'vallekas': 'Villa de Vallecas',
    'seville': 'Sevilla',
    'serranillos del valle': 'Serranillos del Valle',
    'londres': 'London',
    'distrito federal': 'Mexico City',
    'mexico': 'Mexico City',
    'murchante': 'Murchante',
    'viena': 'Vienna',
    'madriz': 'Madrid',
    'huescar': 'Huéscar',
    'catamarca': 'San Fernando del Valle de Catamarca',
    'cabrales': 'Carreña',
    'godelleta': 'Godelleta',
    'almazora': 'Almazora',
    'mollerussa': 'Mollerussa',
    'torrellas': 'Torrellas',
    'madriles': 'Madrid',
    'mollet del valles': 'Mollet del Vallès',
    'compostela': 'Santiago de Compostela',
    'lar': 'Santiago de Compostela',
    'santiago de cali': 'Santiago de Cali',
    'viana do bolo': 'Viana do Bolo',
    'cba': 'Córdoba',
    'boedo': 'Buenos Aires',
    'burzaco': 'Buenos Aires',
    'berazategui': 'Buenos Aires',
    'caballito': 'Buenos Aires',
    'mataderos': 'Buenos Aires',
    'merlo': 'Buenos Aires',
    'trenque lauquen': 'Trenque Lauquen',
    'lanus': 'Lanus',
    'villa urquiza': 'Buenos Aires',
    'bariloche': 'Bariloche',
    'lomas de zamora': 'Buenos Aires',
    'la plata': 'La Plata',
    'nueva york': 'New York City',
    'sonora': 'Sonora'
}

'''
    sinaloa, México
    Manchester, England
    Washington, dc
    San Martín, Argentina
    '''
##
#
# paises

country_dupes_dict={
    'united states': 'United States of America'
}
