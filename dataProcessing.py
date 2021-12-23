import pandas as p
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import feature_extraction
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2

def abc():
    data = pd.read_csv("spam.csv", encoding = "latin-1")
    X = data.iloc[:, 1]  
    y = data.iloc[:,0]  
    y = y.map({"spam":1, "ham":0})  
    features = feature_extraction.text.CountVectorizer(stop_words="english")
    x = features.fit_transform(X);
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(x,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  
    print(featureScores.nlargest(10,'Score'))  


def DataProcessing():
    data = p.read_csv("spam.csv", encoding='latin-1')
    #data.drop_duplicates(inplace= True)
    data["email"].fillna("hello", inplace=True) 
    data["class"].fillna("ham" , inplace=True)

    
    x = data.iloc[:, 1]
    y = data["class"]

    y = y.map({"spam":1, "ham":0})

   # print(y)
    # index = 0;
    # for i in y:
    #     y[index] = 1 if i == "spam" else 0
    #     index += 1
    features = feature_extraction.text.CountVectorizer(stop_words="english")
    x = features.fit_transform(x);
    print(x.shape)

    s = VarianceThreshold(0.05)
    x = s.fit_transform(x)
    print(s.get_feature_names_out())

    

   # print(np.shape(x))
   # print(np.shape(y))
    return x,y

def testing(email):
    email = [email]
    feature = feature_extraction.text.CountVectorizer(stop_words="english")
    email = feature.fit_transform(email)
    print(email.shape)
    print(feature.get_feature_names_out())
    y = [1]
    return email, y



testing(" thank you for shopping with us gifts for all occasions free gift with NUMBER NUMBER purchase for a limited time only receive this NUMBER plush santa bear free with your purchase of NUMBER NUMBER or more when your order totals NUMBER NUMBER or more order must be NUMBER NUMBER or more before shipping and handling this santa bear is added to your cart for free while supplies last mary s store would like to thank you for being a valued customer as our way of saying thanks to you the customer we are offering a NUMBER discount on all purchases made during the month of november just enter the word thanks in the discount code box during checkout to receive your automatic NUMBER discount hyperlink click here hyperlink to enter hyperlink mary s store if you do not wish to receive further discounts please hyperlink click here and type remove in the subject line ")





