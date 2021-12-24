import nltk
import pandas as p
from sklearn import feature_extraction
from csv import DictWriter




from nltk.corpus import stopwords

BagOfWords = list()

def DataProcessing():

    data = p.read_csv("spam.csv", encoding='latin-1')
    #data.drop_duplicates(inplace= True)
    data["email"].fillna("hello", inplace=True) 
    data["class"].fillna("ham" , inplace=True)
    x = data.iloc[:, 1]
    x = x.str.replace("[^a-zA-Z]"," ")
    y = data["class"]
    y = y.map({"spam":1, "ham":0})
    features = feature_extraction.text.CountVectorizer(analyzer= "word", stop_words="english", max_features= 3000)
    x= features.fit_transform(x);
    
    BagOfWords = features.get_feature_names_out()
 
    return x,y,BagOfWords

def  writeNewEmailToCSV(label, email):
    headers = {"class","email"}
    obj = {"class":label,"email":email}
    with open("spam.csv","a",newline="") as fileObject:
        dictWriter = DictWriter(fileObject, fieldnames = headers)
        dictWriter.writerow(obj)
        fileObject.close()





