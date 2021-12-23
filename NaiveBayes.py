from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import normalize
import dataProcessing

from sklearn.metrics import confusion_matrix, accuracy_score

x , y = dataProcessing.DataProcessing()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.50, random_state = 42)

print(X_train.shape)
classifier = MultinomialNB()
classifier.fit(normalize(X_train), y_train)

y_pred = classifier.predict(X_test)
#nx, ny = dataProcessing.testing(" thank you for shopping with us gifts for all occasions free gift with NUMBER NUMBER purchase for a limited time only receive this NUMBER plush santa bear free with your purchase of NUMBER NUMBER or more when your order totals NUMBER NUMBER or more order must be NUMBER NUMBER or more before shipping and handling this santa bear is added to your cart for free while supplies last mary s store would like to thank you for being a valued customer as our way of saying thanks to you the customer we are offering a NUMBER discount on all purchases made during the month of november just enter the word thanks in the discount code box during checkout to receive your automatic NUMBER discount hyperlink click here hyperlink to enter hyperlink mary s store if you do not wish to receive further discounts please hyperlink click here and type remove in the subject line ");



print("acuracy of the model is")
print(accuracy_score(y_test, y_pred))
