import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

data = pd.read_csv("twee150.csv")
data["labels"] = data["class"].map({0: "Abusive comments", 1: "Offensive Language", 2: "No Abusive and Offensive"})
data = data[["comments", "labels"]]

stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

def clean(text):
    text = str(text).lower()
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    # Rest of the cleaning steps...

    return text

data["comments"] = data["comments"].apply(clean)

x = np.array(data["comments"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 15))
plot_tree(clf, feature_names=cv.get_feature_names(), class_names=clf.classes_, filled=True)
plt.show()
