from country_city_sets import clean_string

## Primero un partidor de secuencias y combinaciones (ngrams)
def find_ngrams(input_list):
    ngrams=[]
    for ng_len in range(1,len(input_list)+1):
        for ix in range(1+len(input_list)-ng_len):
            ngrams.append(" ".join(input_list[ix:ix+ng_len]))
    return ngrams

## chequear si un string y sus substrings estan en un conjunto
def check_substring(raw_string, wordlist):
    matches=[]
    string=clean_string(raw_string)
    string_parts = find_ngrams(string.split())
    for substring in wordlist: 
        if substring in string_parts: 
            matches.append(substring)
    return matches