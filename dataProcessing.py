import pandas as p
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn import feature_extraction


def DataProcessing():
    data = p.read_csv("spam.csv", encoding='latin-1')
    data.drop_duplicates(inplace= True)
    data["email"].fillna("hello", inplace=True) 
    data["class"].fillna("ham" , inplace=True)

    
    x = data.iloc[:, 1].values
    y = data.iloc[:, 0].values
    index = 0;
    for i in y:
        y[index] = 1 if i == "spam" else 0
        index += 1

    features = feature_extraction.text.CountVectorizer(stop_words="english")
    x = features.fit_transform(x);

    return x,y

   

x, y = DataProcessing()





