
# coding: utf-8

import pandas as pd
pd.set_option("display.max_columns",999)
import numpy as np
import json

# # Getting some info from JSON file

# ## Data set preparation, run everything to get from scratch

def extractLoc(tweet,root=''):
    if root!='':
        root=root+'_' # si hay algo apendarle _
    tweetLoc={}
    if 'user' in tweet and 'location' in tweet['user']:
        tweetLoc[root+'location']=tweet['user']['location']
        tweetLoc[root+'time_zone']=tweet['user']['time_zone']
    else:
        tweetLoc[root+'location']=None
        tweetLoc[root+'time_zone']=None
    # try to geolocate the tweet
    if 'coordinates' in tweet and tweet['coordinates']!=None:
        tweetLoc[root+'lat']=tweet['coordinates']['coordinates'][0]
        tweetLoc[root+'lon']=tweet['coordinates']['coordinates'][1]
    else:
        tweetLoc[root+'lat']=None
        tweetLoc[root+'lon']=None
        
    if 'place' in tweet and tweet['place']!=None:
        tweetLoc[root+'country']=tweet['place']['country']
        tweetLoc[root+'place']=tweet['place']['full_name']
    else:
        tweetLoc[root+'country']=None
        tweetLoc[root+'place']=None 
    return tweetLoc

def extractUser(tweet,root=''):
    if root!='':
        root=root+'_' # si hay algo apendarle _
    tweetUser={}
    attrs_to_extract = ['id','name','screen_name','followers_count','statuses_count','created_at']
    for attr in attrs_to_extract:
        if 'user' in tweet and attr in tweet['user']:
            tweetUser[root+'user_'+attr]=tweet['user'][attr]  
        else:
            tweetUser[root+'user_'+attr]=None
    return tweetUser

def extractGeneralInfo(tweet,root=''):
    if root!='':
        root=root+'_' # si hay algo apendarle _
    tweetInfo={}
    attrs_to_extract = ['id','retweet_count','favorite_count','full_text','quote_count','created_at']
    for attr in attrs_to_extract:
        tweetInfo[root+attr]=tweet[attr] if attr in tweet else None
    return tweetInfo
        
# info from https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
def extractInfo(tweet):
    tweetInfo={}

    # basic tweet info
    tweetInfo.update(extractGeneralInfo(tweet))
    # get tweet location
    tweetInfo.update(extractLoc(tweet))
    # This tweet user data
    tweetInfo.update(extractUser(tweet))

    # type of tweet and get parent tweet included
    if tweet['in_reply_to_status_id']!=None: # reply
        tweetInfo['tweet_type']='reply'
        subtweet={ 'id': tweet['in_reply_to_status_id'], 'user': {'id': tweet['in_reply_to_user_id']}}        
    elif 'quoted_status' in tweet: # quote
        tweetInfo['tweet_type']='quote'
        subtweet = tweet['quoted_status']
        subtweet['id']=tweet['quoted_status_id']                      
    elif'retweeted_status' in tweet and tweet['retweeted_status']!=None: # retweet
        tweetInfo['tweet_type']='retweet'
        subtweet = tweet['retweeted_status']
    else: # iriginal
        tweetInfo['tweet_type']='original'
        subtweet={}
          
    # get subtweet  data
    tweetInfo.update(extractGeneralInfo(subtweet,'parent'))
    # get subtweet user data
    tweetInfo.update(extractUser(subtweet,'parent'))    
    # get subtweet location
    tweetInfo.update(extractLoc(subtweet,'parent'))
                    
    return tweetInfo

tweets={}
tweetfile=open("../data/Aniol-Maria-cuentalo-search-20180427_20180513.jsonl")
for idx,line in enumerate(tweetfile):
    if idx>100000000: # cap for testing
        break
    tweet=json.loads(line)
    tweets[tweet['id']]=extractInfo(tweet)
tweetfile.close()
print('main data:',len(tweets))


df=pd.DataFrame(tweets).transpose()

#generar fichero
df.to_pickle("../pickles/cuentalo_json_to.pkl")


tweets={}

tweetfile=open("../data/cuentalo_faltantes.jsonl")
for idx,line in enumerate(tweetfile):
    if idx>100000000: # use for testing
        break
    tweet=json.loads(line)
    tweets[tweet['id']]=extractInfo(tweet)
    if tweets[tweet['id']]['tweet_type']=='retweet':
        print (tweets[tweet['id']]['parent_id'])

print('first addendum:',len(tweets))
tweetfile.close()

df=pd.DataFrame(tweets).transpose()

#generar fichero
df.to_pickle("../pickles/cuentalo_json_to_extra_1.pkl")

tweets={}
tweetfile=open("../data/cuentalo_faltantes_2.jsonl")
for idx,line in enumerate(tweetfile):
    if idx>100000000: # use for testing
        break
    tweet=json.loads(line)
    tweets[tweet['id']]=extractInfo(tweet)
    if tweets[tweet['id']]['tweet_type']=='retweet':
        print (tweets[tweet['id']]['parent_id'])

print('2nd addendum:',len(tweets))
tweetfile.close()

df=pd.DataFrame(tweets).transpose()

#generar fichero
df.to_pickle("../pickles/cuentalo_json_to_extra_2.pkl")




