from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import dataProcessing

from sklearn.metrics import confusion_matrix, accuracy_score

x , y = dataProcessing.DataProcessing()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("acuracy of the model is")
print(accuracy_score(y_test, y_pred))
