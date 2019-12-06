from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
import numpy as np
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os
from master import builder

qwerty = 3 #the number of fold used for cross validation
alpha_value = 0.1

def neurronal(days,period,qwerty,alpha_value):
    if os.path.exists("Built_Data/built_movies_days_%s_period_%s"%(days,period)) == True:
        built_movies = pd.read_csv("Built_Data/built_movies_days_%s_period_%s"%(days,period),sep = ",")
    else:
        builder(period,days)
        built_movies= pd.read_csv("Built_Data/built_movies_days_%s_period_%s"%(days,period),sep = ",")

    #columns in .csv output file initialization:
    classifier = []
    alpha_column = []
    layer_structure = []
    days_parsing_comments = []
    prediction_period = []
    cross_validation_fold = []
    score_average = []
    score_std = []


    #Ordinal Value per date
    ordinal_date=[]
    for row in built_movies["Trailer Date"]:
        row=row.split("(")[1]
        row=row.split(")")[0]
        year = int(row.split(",")[0])
        month = int(row.split(",")[1])
        day = int(row.split(",")[2])
        row = ordinal_date.append(datetime.date(year,month,day).toordinal())
    ordinal_date = [i - min(ordinal_date) for i in ordinal_date ]
    built_movies["Ordinal Dates"] = ordinal_date

    #Ordinal Value per Studio
    ordinal_studio=[]
    for row in built_movies["TICKER"]:
        if row == "CMCSA":
            ordinal_studio.append(1)
        elif row == "DIS":
            ordinal_studio.append(2)
        elif row == "LGF":
            ordinal_studio.append(3)
        elif row == "SNE":
            ordinal_studio.append(4)
        elif row == "TWX":
            ordinal_studio.append(5)
        elif row == "VIA":
            ordinal_studio.append(6)
        else:
            #print("A movie Ticker is not recognized, we have inputed a 0 in place (；一_一)")
            ordinal_studio.append(0)
    built_movies["Ordinal TICKER"] = ordinal_studio


    #Splitting the data set in a target set and a non-target set
    X = built_movies.drop(["Trailer YT_id","Period_price_Variation","MOVIE","TICKER","Trailer Date","Categorized Price Variation"],axis=1).values
    y = built_movies["Categorized Price Variation"].values #.values permit to create a list of the data
    #y = built_movies["Period_price_Variation"].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=53)

    #Scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    #outputing a global score file with previously initialized columns
    for premier in range(1,11):
        print("Sequence :%s"%premier)
        a = premier * 10
        for second in range(1,11):

            b = second * 10
            for troisieme in range(1,11):
                c = troisieme * 10

                #Initialazing neural classifier
                clf = MLPClassifier(solver='lbfgs', alpha=alpha_value,hidden_layer_sizes=(a,b,c), random_state=1)
                clf.fit(X_train,y_train)
                scores = cross_val_score(clf,X_test,y_test,cv=qwerty)

                score_average.append(scores.mean())
                score_std.append(scores.std())
                days_parsing_comments.append(days)
                prediction_period.append(period)
                cross_validation_fold.append(qwerty)
                layer_structure.append((a,b,c))
                alpha_column.append(alpha_value)
                classifier.append("MLPClassifier")

    output = pd.DataFrame(list(zip(classifier,alpha_column,cross_validation_fold,layer_structure,prediction_period,days_parsing_comments,score_std,score_average)),
    columns = ["Classifier","Alpha Value","Cross_Val folds","Layer Structure","Prediction Period","Parsing Day range","Score Std","Score average"])
    print(output)
    output.to_csv("Results/MLPClassifier_score_list.csv",index=False,mode='a')


#looking through several combinations to get the best schedule for parameters time and period

#looking of score value with one day difference between days and period
for days in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]:
    period = days + 1
    neurronal(days,period,qwerty,alpha_value)

#looking of score value with two days difference between days and period
for days in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]:
    period = days + 3
    neurronal(days,period,qwerty,alpha_value)

#looking of score value with two days difference between days and period
for days in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]:
    period = days + 5
    neurronal(days,period,qwerty,alpha_value)

#looking of score values with 10 days difference between days and period
for days in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]:
    period = days + 10
    neurronal(days,period,qwerty,alpha_value)

#testing longer period range to see long term prediction possibilities
for period in [50,60,70,100,130,160,180,200]:
    days = 20
    neurronal(days,period,qwerty,alpha_value)
