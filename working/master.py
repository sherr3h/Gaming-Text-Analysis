import pandas as pd
import datetime
from datetime import timedelta
import re
import os
from Parser import info
from Parser import parser
from Parser_Financial import parser_financial
from Parser_Financial import price_variation





period = input("PLS enter the timeframe on which you wish to predict a price variation:")
days = input("PLS enter the timeframe on which you wish to parse comments:")

period,days = int(period),int(days)




#From comments
polarity_column = [] #include the average polarity of comments under a video
subjectivity_column = []  #include the average subjectivity of comments under a video
polarity_StdDeviation_column = [] #include the Standard Deviation of comments polarity under a video
subjectivity_StdDeviation_column = [] #include the Standard Deviation of comments subjectivity under a video

films = []
Trailer_date = []
studios = []


#From financial_parser
previous_month_price_variation = []
period_price_variation = [] #It is the Future price evolution, what we want to predict
average_volume = []
month_volatility = []
#From overall.csv
like_count = []
dislike_count = []
views = []
comment_count = []


liste =[i for i in os.listdir("Raw_Data_Studios") if re.match(".DS_S",i) == None]
#Gather the data to build the finale Dataframe
for TICKER in liste:
    parser(days,TICKER) #ensure the pre-processed data is present
    for movie in os.listdir("Data_Studios/%s/Studio_%s_%s"%(TICKER,TICKER,days)):
            df = pd.read_csv("Data_Studios/%s/Studio_%s_%s/%s"%(TICKER,TICKER,days,movie),lineterminator='\n')
            df = pd.DataFrame(df)
            print("Master "+movie +" "+ TICKER)
            #few options to link the data from comments to overal_comments.csv, we link the two data file per video id
            overall = pd.read_csv("overall_comments_24_nov_CLEAN.csv",lineterminator='\n',sep=",")
            overall = pd.DataFrame(overall)
            id = df.iloc[0,1]


            #append the list from data which comes from data
            for index,row in overall.iterrows():
                if id  == row[1]:
                    like_count.append(row[4])
                    dislike_count.append(row[5])
                    views.append(row[3])
                    comment_count.append(row[-4])


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
                average_volume.append(price_variation(period,TICKER,movie)[2])
                month_volatility.append(price_variation(period,TICKER,movie)[3])
            except:
                 print("COULD NOT HAVE FINANCIAL DATA %s"%movie)
                 previous_month_price_variation.append(0)
                 period_price_variation.append(0)
                 average_volume.append(0)
                 month_volatility.append(1)




output = pd.DataFrame(list(zip(films,studios,Trailer_date,like_count,dislike_count,views,comment_count, polarity_column,subjectivity_column,polarity_StdDeviation_column,
subjectivity_StdDeviation_column,previous_month_price_variation,average_volume,month_volatility,
period_price_variation)), columns =["MOVIE","TICKER","Trailer Date","Like","Dislike","Views","Comments","Polarity","Subjectivity","Polarity_Std","Subjectivity_Std",
"Previous_month_price_variation","Average Volume","Volatility","Period_price_Variation"])

print(output)
output.to_csv("petitPoucet1.csv", index = False)
