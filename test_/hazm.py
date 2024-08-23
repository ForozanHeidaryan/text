import numpy as np
from hazm import*
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_theme(style="darkgrid")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import hazm
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

df=pd.read_excel('/media/forozan/1A6451126450F24D/vsToken/sampleD.xlsx')
df['Category'] = df['Category'].astype('category')

#delete duplicate rows in data_set

def duplicate_rows(data):
    data = data.drop_duplicates(keep = 'first')
    return data 

df = duplicate_rows(df)

#remove NaN Value

df = df.dropna()

#remove space from Category columns

def remove_spaces(string):
    return string.strip()

'''
#for other Category'
df[] = df[].apply(remove_spaces)

'''
'''
#encodin for cols category 

def encode_col(data , cols):
    encoder = LabelEncoder()
    for col in cols:
        def[col] = encoder.fit_transform(data[col])
        return data

columns_to_encode =['' , '']

df = encode_col(df , columns_to_encode) '''


#visualization Class Distribtion using a Bar Chart
def plot_class_distribution(data):
    class_counts = data['Category'].value_counts()
    class_labels = class_counts.index
    
    fig , ax = plt.subplots()
    ax.bar(class_labels , class_counts , lw=20,color= 'green')
    
    ax.set_xlabel('Category')
    ax.set_title('Count')
    ax.set_ylabel('Class Distribtion')
    
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()    

plot_class_distribution(df)

# Preprocessing Persian Text Using Hazm Library

def preprocessing_text(text):
    # Remove non-Persian characters
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)
    
    # Tokenize
    tokenizer = hazm.word_tokenize
    tokens = tokenizer(text)
    
    # Lemmatization
    lemmatizer = hazm.Lemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Removal of stopwords
    stopwords = hazm.stopwords_list()
    filtered_tokens = [token for token in lemmatized_tokens if token not in stopwords]
    filtered_sentence = ' '.join(filtered_tokens)
    
    return filtered_sentence
import time
start_time = time.time()
df['Title'] = df['Title'].apply(preprocessing_text)
print('Fit time : ', time.time() - start_time)


# Text Vectorization using TF-IDF for the 'title' Column
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vec_title = tfidf_vectorizer.fit_transform(df.Title)

X = df['Title']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Training an XGBoost Text Classification Model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(x_train)

# Calculate F1-score for training
f1 = f1_score(y_train, y_pred, average='weighted')

print("F1-score:", f1)