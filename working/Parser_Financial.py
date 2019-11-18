import pandas as pd
import os
from datetime import datetime
import Parser
from datetime import timedelta


#the aim of this code is to build the csv sheet input to develop machine learning
if __name__ == "__main__":
    days = input("Day range for comment:")
    TICKER = input("Specify Ticker:")
    period = input("Specify period of stock variation you want to observe:") #for which period are you looking for price variation
    period,days = int(period),int(days)
else:
    print("")


def parser_financial(period,TICKER,days):
    output = []
    df_fin = pd.read_csv("Financial DATA/%s.csv"%TICKER)
    df_fin = pd.DataFrame(df_fin)

    liste = ["Raw_Data_Studios/%s/"%TICKER+i for i in os.listdir("Raw_Data_Studios/%s"%TICKER)]
    date_ba = Parser.info(days,liste,TICKER)[0] - timedelta(days=31) #parser input are the day range you want to grab after the release of the video and you need to input a list of video's comments to parse
    b = date_ba+ timedelta(days = period)

    for index,row in df_fin.iterrows():
        row[0] = datetime.strptime(row[0], '%Y-%m-%d' )
        if (row[0] > date_ba) & (row[0]<b):
            output.append(row)
        else:
            continue

    if os.path.exists("Data_Studios/%s"%TICKER) == False: #those if statement are just here to build the directories where the data will be droped
        os.mkdir("Data_Studios/%s"%TICKER)
        print("PRINTING %s period%s"%(TICKER,period))
        output = pd.DataFrame(output)
        output.to_csv("Data_Studios/%s/Fin_data_%s_period_%s.csv"%(TICKER,period,TICKER),index = False)
    else:
        if os.path.exists("Data_Studios/%s/Fin_data_%s_period_%s.csv") == True:
            print("The directory for financial data on this period is already here therefore we won't parse it again")
        else:
            print("PRINTING %s period%s"%(TICKER,period))
            output = pd.DataFrame(output)
            output.to_csv("Data_Studios/%s/Fin_data_%s_period_%s.csv"%(TICKER,period,TICKER),index = False)


    output = pd.DataFrame(output)
    output.to_csv("Data_Studios/%s/Fin_data_%s_period_%s.csv"%(TICKER,period,TICKER),index = False)

#parser_financial(period = period,TICKER = TICKER,days = days)
