import pandas as pd
import datetime
from datetime import timedelta
import re
import os
from Parser import parser
from Parser_Financial import parser_financial
from textblob import TextBlob #the master code mnake the sentiment analysis himslef as it is concerned with data interpretation

#the goal is to build the program which will manage the workflow between parser_financial and parser (which take the comments)

#il faut déjà faire une colonne avec les sentiments, un collone avec la price variation sur 1 mois un avant la vidéo, et la price variation
#après la période écoulée.
TICKER = "DIS"
period = 25
days = 5
liste = [i for i in os.listdir("Raw_Data_Studios/%s"%TICKER) if re.match("overall",i) == None] #a penser A INTRODUIRE DANS PARSER.PY A LAEVENIR,trouver la liste selon le TICKER
#liste = [i for i in liste if re.match("overall",i) == None]
period,days = int(period),int(days)
colonne_sentiments=pd.DataFrame()


#parser(days,liste,TICKER)#faut mettre les variables plus haut avec des input qu'on gardera comme dans financial parser, il y avait des méthodes pour cela dans le code youtube api
#parser_financial(period,TICKER,days)
#colonne_sentiments = pd.DataFrame([])
for TICKER in os.listdir("Data_Studios"):
    for movie in os.listdir("Data_Studios/"+TICKER+"/Studio_%s_%s"%(TICKER,days)):
        try:
            print("Data_Studios/"+TICKER+"/Studio_%s_%s/%s"%(TICKER,days,movie))
            df = pd.read_csv("Data_Studios/%s/Studio_%s_%s/%s"%(TICKER,TICKER,days,movie))
            df = pd.DataFrame(df)["Comment"]

        except:
            #pandas.errors.EmptyDataError: No columns to parse from file
            print("Data_Studios/"+TICKER+"/Studio_%s_%s/%s is EMPTY"%(TICKER,days,movie))
            continue

        l_blob_list = [TextBlob(str(ligne)).sentiment.polarity for ligne in df ]
        colonne_sentiments["%s"%movie] = l_blob_list


        #On fait un DataFrame qui reprends par films du studio la liste
        #des sentiments par commentaires, et ensuite on append ce DataFrame["Titre_du_film"] de
        #chaque commentaire sentiment

        #colonne_sentiments["%s"%movie].append(l_blob) OLD CODE

colonne_sentiments.to_csv("colonne_sentiment.csv", index = False)
