# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #tab delimeter and 3 stands for avoiding quating

#Cleaning the texts
import re
import nltk
nltk.download('stopwords') #remowing excaimations and adv
from nltk.corpus import stopwords #remowing excaimations and adv
from nltk.stem import PorterStemmer #stemming
corpus = [] #corpus is the collection of text
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'] [i]) #extracting words only
    review = review.lower() #subatituting capital letters
    review = review.split() #splitting the string
    ps = PorterStemmer() #stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) #joining the words into one string
    corpus.append(review)
    
#Creating the Bag for Words model (sparse matrix)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values

#Splitting the dataset into the training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)


#Putting the values in the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Classifier to the training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, Y_train)

#Prdicting the test set results
Y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred) 
