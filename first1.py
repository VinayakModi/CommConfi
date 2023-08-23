import gradio as gr
#gradio for buliding GUI
import pandas as pd
#pandas for reading and futher calculation
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string

# Load the dataset
data = pd.read_csv("commentdatset.csv")
data["labels"] = data["class"].map({0: "Offensive Language", 1: "Abusive comments", 2: "No Abusive and Offensive"})
data = data[["comments", "labels"]]

# Download NLTK resources
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# Clean data
def clean(text):
    text = str(text).lower()
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
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

# Function to classify comments
def classify_comment(comment):
    cleaned_comment = clean(comment)
    vectorized_comment = cv.transform([cleaned_comment])
    prediction = clf.predict(vectorized_comment)
    return prediction[0]

# Create the input and output interfaces
comment_input = gr.inputs.Textbox(label="Enter a comment")
classification_output = gr.outputs.Label(num_top_classes=1)

# Create the Gradio interface
interface = gr.Interface(fn=classify_comment, inputs=comment_input, outputs=classification_output, title="Comment Classifier")
interface.launch()
