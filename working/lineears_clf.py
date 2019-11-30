from sklearn import svm
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

built_movies = pd.read_csv("petitPoucet1.csv",sep = ",")

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
        print("A movie Ticker is not recognized, we have inputed a 0 in place (；一_一)")
        ordinal_studio.append(0)
built_movies["Ordinal TICKER"] = ordinal_studio

#Splitting the data set in a target set and a non-target set
X = built_movies.drop(["Trailer YT_id","Period_price_Variation","MOVIE","TICKER","Trailer Date","Categorized Price Variation",],axis=1).values
y = built_movies["Period_price_Variation"].values #.values permit to create a list of the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ridge = Ridge(alpha=0.01,normalize=True).fit(X_train,y_train)
# knn = KNeighborsClassifier(n_neighbors = 6).fit(X_train,y_train)
# print("Ridge Score: "+ridge.score(X_test,y_test))
# print("KNN Score: "+ridge.score(X_test,y_test))



for reg in regression_models:
    print(regression_models)
    reg.fit(X_train,y_train)
    print(reg.score(X_test,y_test))
