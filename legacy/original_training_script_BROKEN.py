# Importing the libraries
'''Customer review feedback project'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv(r'D:\AI NLP -NATURAL LANGUAGE PROCESSING DATA\Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)



# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values


''' Convert from nlp technique to number'''


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)
'''


# it is give a correct model
from sklearn.ensemble import RandomForestClassifier
classifier =RandomForestClassifier(max_depth=4, n_estimators=60, random_state=0,criterion='entropy')
classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


#Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_pred,y_test)
print(cm)

#Accurecy
from sklearn.metrics import accuracy_score
ac =accuracy_score(y_pred,y_test)
print(ac)

bias =classifier.score(X_train,y_train)
bias

variance =classifier.score(X_test,y_test)
variance


#Confusion matrix Visualize
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

#train and test bar
plt.figure(figsize=(4, 4))
bars = plt.bar(['Train', 'Test'], [train_acc, test_acc],
               color=['#3B6D11', '#185FA5'], width=0.4)
for i, v in enumerate([train_acc, test_acc]):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
plt.ylim(0, 110)
plt.ylabel('Accuracy (%)')
plt.title('Train vs Test Accuracy')
plt.tight_layout()
plt.show()



plt.figure(figsize=(5, 4))
plt.hist(y_prob[y_test_int == 1], bins=25, alpha=0.6, color='green', label='Positive')
plt.hist(y_prob[y_test_int == 0], bins=25, alpha=0.6, color='red',   label='Negative')
plt.axvline(0.5, color='black', linestyle='--', label='Threshold')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Prediction Probability Distribution')
plt.legend()
plt.tight_layout()
plt.show()
