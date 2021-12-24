from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import normalize
import dataProcessing

from sklearn.metrics import accuracy_score

x , y , BagOfWords= dataProcessing.DataProcessing()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.50, random_state = 42)

print(X_train.shape)
classifier = MultinomialNB()
classifier.fit(normalize(X_train), y_train)

y_pred = classifier.predict(X_test)
print("acuracy of the model is")
print(accuracy_score(y_test, y_pred))

email = "it has come to our attention that your citizensr bank account information needs to be updated as part of our continuing commitment to protect your account and to reduce the instance of fraud on our website . if you could please take 5 - 10 minutes out of your online experience and renew your records you will not run into any future problems with the online service . however , failure to confirm your records may result in your account suspension .you have confirmed your account records your internet banking service will not be interrupted and will continue as normal .to confirm your bank account records please click here .note :this e - mail was sent on behalf of the online banking community , if you do not have an online banking account with charterr one then this message does not apply to you and you may ignore this message .thank you for your time ,citizensr financial group ."
email1 = email
email = [email]
features = feature_extraction.text.CountVectorizer(vocabulary=BagOfWords)
email = features.fit_transform(email)
Y = classifier.predict(email)

if(Y[0] == 1):
    print("the given email is spam")
else:
    print("the given email is not spam")




