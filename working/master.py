import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
import re
import os
from Parser import info
from Parser import parser
from Parser_Financial import parser_financial
from Parser_Financial import price_variation
from textblob import TextBlob



TICKER = input("PLS enter the TICKER:")
period = input("PLS enter the timeframe on which you wish to predict a price variation:")
days = input("PLS enter the timeframe on which you wish to parse comments:")

period,days = int(period),int(days)
master = pd.DataFrame()

films = []
Trailer_date = []
studios = []

#From comments
polarity_column = [] #include the average polarity of comments under a video
subjectivity_column = []  #include the average subjectivity of comments under a video
polarity_StdDeviation_column = [] #include the Standard Deviation of comments polarity under a video
subjectivity_StdDeviation_column = [] #include the Standard Deviation of comments subjectivity under a video

#From financial_parser
previous_month_price_variation = []
period_price_variation = [] #It is the Future price evolution, what we want to predict

#From overall.csv
like_count = []
dislike_count = []
views = []

#ensure the necessary data is present and already pre_processed
parser(days,TICKER)

#Gather the data to build the finale Dataframe
for TICKER in os.listdir("Data_Studios"):
    for movie in os.listdir("Data_Studios/%s/Studio_%s_%s"%(TICKER,TICKER,days)):
            df = pd.read_csv("Data_Studios/%s/Studio_%s_%s/%s"%(TICKER,TICKER,days,movie),lineterminator='\n')
            df = pd.DataFrame(df)



            for index,row in df.iterrows():
                row[-2] = float(row[-2])
                row[-1] = float(row[-1])

            films.append(movie.split("official")[0])

            polarity_column.append(df["polarity"].mean(axis = 0,skipna = True))
            subjectivity_column.append(df["subjectivity"].mean(axis = 0,skipna = True))
            polarity_StdDeviation_column.append(df["polarity"].std(axis = 0,skipna = True))
            subjectivity_StdDeviation_column.append(df["subjectivity"].std(axis = 0,skipna = True))
            Trailer_date.append(info(TICKER,movie))
            studios.append(TICKER)
            try:
                previous_month_price_variation.append(price_variation(period,TICKER,movie)[0])
                period_price_variation.append(price_variation(period,TICKER,movie)[1])
            except:
                print("COULD NOT HAVE FINANCIAL DATA %s"%movie)
                previous_month_price_variation.append(0)
                period_price_variation.append(0)



output = pd.DataFrame(list(zip(films,studios,Trailer_date, polarity_column,subjectivity_column,polarity_StdDeviation_column,
subjectivity_StdDeviation_column,previous_month_price_variation,
period_price_variation)), columns =["MOVIE","TICKER","Trailer Date","Polarity","Subjectivity","Polarity_Std","Subjectivity_Std",
"Previous_month_price_variation","Period_price_Variation"])

print(output)
output.to_csv("CeciEstUnExemple2.csv", index = False)
