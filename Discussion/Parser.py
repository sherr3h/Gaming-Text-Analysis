import pandas as pd
from datetime import timedelta
from datetime import datetime
import os
from textblob import TextBlob
import re


#to run the code you need to build a directory "financial data" where you identify drop each studio TICKERS financial data from Yahoo without changing the filename
#You also need to make a directory "Raw_Data_Studios" where you drop each of your studios downloaded comments in folders that you will call by their respectives TICKERS
#Finally, create a directory "Data_Studios" that you will let emplty



#days = input("Number of days for the comments selection:")
#days = int(journées)
days = 5
TICKER = "DIS"
liste = [i for i in os.listdir("Raw_Data_Studios/%s"%TICKER) if re.match("overall",i) == None]

def parser(days,liste,TICKER):
    for filename in liste:
        #filename_output = filename.split("/")[1] #this is one is to use if you use the code directly,when you don't iterate it through an external list
        print(filename) #ten times to notice it easily
        df = pd.read_csv("Raw_Data_Studios/DIS/Indestructibles 2official trailer_comments.csv",sep = ",")
        output = []
        sent_s = []
        sent_p = []
        date_ba = df.iloc[0,0].split('Z')[0]
        date_ba = datetime.strptime(date_ba, '%Y-%m-%dT%H:%M:%S.%f')
        b = date_ba + timedelta(days = days)

        if os.path.exists("Data_Studios/%s/Studio_%s_%s/%s.csv"%(TICKER,TICKER,days,filename)) == False:
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
            output.to_csv("Data_Studios/%s/Studio_%s_%s/%s.csv"%(TICKER,TICKER,days,filename), index=False)

        else:
            continue




def info(days, liste, TICKER):
    for filename in liste:
        df = pd.read_csv(filename)
        date_ba = df.iloc[0,0].split('Z')[0]
        date_ba = datetime.strptime(date_ba, '%Y-%m-%dT%H:%M:%S.%f')
        b = date_ba + timedelta(days = days)

        c = [date_ba,b]

        return c

parser(days,liste,"DIS")
