import pandas as pd
from datetime import timedelta
from datetime import datetime
import os
from textblob import TextBlob
import re




def parser(days,TICKER):
    liste = [i for i in os.listdir("Raw_Data_Studios/%s"%TICKER) if (re.match("overall",i) == None) & (re.match(".DS_Store",i)== None)]
    insuficent_data_movies = []
    for movie in liste:
        #filename_output = filename.split("/")[1] #this is one is to use if you use the code directly,when you don't iterate it through an external list
        print("Parser "+movie +" "+ TICKER)
        movie_court = movie.split("/")[-1]
        if os.path.exists("Data_Studios/%s/Studio_%s_%s/%s"%(TICKER,TICKER,days,movie_court)) == False:
            df = pd.read_csv("Raw_Data_Studios/%s/%s"%(TICKER,movie),sep = ",",lineterminator='\n')
            output = []
            sent_s = []
            sent_p = []
            date_ba = df.iloc[0,0].split('Z')[0]
            date_ba = datetime.strptime(date_ba, '%Y-%m-%dT%H:%M:%S.%f')
            b = date_ba + timedelta(days = days)


            for index,row in df.iterrows():
                row[-2] = str(row[-2])
                row[-2] = datetime.strptime(row[-2].split('Z')[0], '%Y-%m-%dT%H:%M:%S.%f' )
                if (row[-2] > date_ba) & (row[-2]<b):
                    output.append(row)
                    try:
                        polarité = TextBlob(row[-3]).sentiment.polarity
                        sent_p.append(polarité)
                        subjectivité = TextBlob(row[-3]).sentiment.subjectivity
                        sent_s.append(subjectivité)

                    except:
                        print("Missing value detected")
                        print(row)
                        polarité = 0
                        subjectivité = 0
                        sent_p.append(polarité)
                        sent_s.append(subjectivité)
                        continue
                else:
                    continue

            if os.path.exists("Data_Studios/%s"%TICKER) == False: #those if statement are just here to build the directories where the data will be droped
                os.mkdir("Data_Studios/%s"%TICKER)
                if os.path.exists("Data_Studios/%s/Studio_%s_%s"%(TICKER,TICKER,days)) == False:
                    os.mkdir("Data_Studios/%s/Studio_%s_%s"%(TICKER,TICKER,days))
            else:
                if os.path.exists("Data_Studios/%s/Studio_%s_%s"%(TICKER,TICKER,days)) == False:
                    os.mkdir("Data_Studios/%s/Studio_%s_%s"%(TICKER,TICKER,days))

            output = pd.DataFrame(output)
            output["polarity"] = pd.Series(sent_p,index=output.index)
            output["subjectivity"] = pd.Series(sent_s,index=output.index)
            if len(output) > 5:
                output.to_csv("Data_Studios/%s/Studio_%s_%s/%s"%(TICKER,TICKER,days,movie), index=False)
            else:
                print("No comments matching the date range inputed for %s"%movie)
                insuficent_data_movies.append(movie)

        else:
            continue

    insuficent_data_movies = pd.DataFrame(insuficent_data_movies)
    insuficent_data_movies.to_csv("Movies with insuficent data.csv",index = False,sep=",")




def info(TICKER,movie):
    movie_court = movie.split("/")[-1]
    #df = pd.read_csv(movie,sep = ",",lineterminator = "\n")
    df = pd.read_csv("Raw_Data_Studios/%s/%s"%(TICKER,movie_court))
    date_ba = df.iloc[0,0].split('Z')[0]
    date_ba = datetime.strptime(date_ba, '%Y-%m-%dT%H:%M:%S.%f')

    c = [date_ba]

    return c




if __name__ == "__main__":
    days = input("days=")
    days = int(days)
    TICKER = input("TICKER=")
    parser(days,TICKER)
