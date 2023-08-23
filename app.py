from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("commentdatset.csv")
data["labels"] = data["class"].map({0: "Abusive comments", 1: "Offensive Language", 2: "No Abusive and Offensive"})
data = data[["comments", "labels"]]

# Download NLTK resources
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")


# Clean data
def clean(text):
    text = str(text).lower()
    text = re.sub(r"she's", "she is", text)
    # Rest of the cleaning code...
    return text


data["comments"] = data["comments"].apply(clean)

# Extract features
x = np.array(data["comments"])
y = np.array(data["labels"])
cv = CountVectorizer()
X = cv.fit_transform(x)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# Define API endpoint
@app.route('/classify', methods=['POST'])
def classify_comment():
    # Get comment from request data
    comment = request.json['comment']

    # Clean the comment
    cleaned_comment = clean(comment)

    # Vectorize the comment
    vectorized_comment = cv.transform([cleaned_comment])

    # Perform classification
    prediction = clf.predict(vectorized_comment)

    # Get the predicted label
    predicted_label = prediction[0]

    # Return the result as JSON
    result = {'comment': comment, 'predicted_label': predicted_label}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
