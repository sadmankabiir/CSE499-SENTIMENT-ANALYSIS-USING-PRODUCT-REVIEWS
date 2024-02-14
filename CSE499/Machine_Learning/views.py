from django.shortcuts import render
# from django.http import HttpResponse
# import pickle
# import numpy as np
# import pandas as pd
# import sklearn
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk import pos_tag, word_tokenize
# from collections import Counter

from .apps import MachineLearningConfig
from . forms import UserInput

# def hlowerCase(text):
#     text = text.lower()
#     return text

# def catergorize(row):
#     if row['Star Rating'] >= 0 and row['Star Rating'] <= 2:
#         return -1
#     elif row['Star Rating'] == 3:
#         return 0
#     elif row['Star Rating'] >= 4 and row['Star Rating'] <= 5:
#         return 1
#     else:
#         return 100

# def removestop(text):
#     text = [word for word in text.split() if word not in stopwords]
#     return text

# # Create your views here.
# # def userinput(request):
# #     return render(request, 'userInput.html')

# def result(userStr):
#     df = pd.read_csv('amazondata.csv')
#     global stopwords
#     df.head()
#     newDf = df[['Review Body', 'Review Headline', 'Star Rating']]
#     newDf.head()
#     newDf[['Star Rating']] = newDf[['Star Rating']].fillna(0)
#     ratings = newDf[newDf['Star Rating'] > 5].index
#     newDf.drop(ratings, inplace=True)
#     newDf['Sentiment'] = newDf.apply(lambda row: catergorize(row), axis=1)
#     newDf['Review Headline'] = newDf['Review Headline'].astype(str)
#     newDf['Review Headline'] = newDf['Review Headline'].apply(hlowerCase)
#     df1 = newDf.drop('Review Body', axis=1)
#     stopwords = set(stopwords.words('english'))
#     df1['Review Headline'] = df1['Review Headline'].apply(removestop)
#     df1['Review Headline'] = df1['Review Headline'].apply(
#         lambda a: ' '.join(a))
#     Counter(' '.join(df1['Review Headline']).split()).most_common(100)
#     morewords = {'stars', '...', 'story', '-', 'series', 'book!', 'books',
#                 '0:00', 'life', 'history', 'novel', 'it!', 'it.', '&', '0', '.', "i'm"}
#     stopwords.update(morewords)
#     df1['Review Headline'] = df1['Review Headline'].apply(removestop)
#     df1['Review Headline'] = df1['Review Headline'].apply(
#         lambda a: ' '.join(a))
#     Counter(' '.join(df1['Review Headline']).split()).most_common(100)
#     morewords = {'five', 'one', 'four', 'three', 'first', 'two', 'characters', 'read.', 'author', 'read!', 'want',
#                 'people', 'work', 'review', 'every', 'know', 'war', 'end', "i've", '1', '2', 'family', 'historical', '3'}
#     stopwords.update(morewords)
#     df1['Review Headline'] = df1['Review Headline'].apply(removestop)
#     df1['Review Headline'] = df1['Review Headline'].apply(
#         lambda a: ' '.join(a))
#     Counter(' '.join(df1['Review Headline']).split()).most_common(100)
#     morewords = {'book', 'another', 'book.', 'time', 'star',
#                 'book,', 'old', "can't", 'years', 'go', '5', 'man', 'ok'}
#     stopwords.update(morewords)
#     df1['Review Headline'] = df1['Review Headline'].apply(removestop)
#     df1['Review Headline'] = df1['Review Headline'].apply(
#         lambda a: ' '.join(a))
#     Counter(' '.join(df1['Review Headline']).split()).most_common(100)
#     X_train_raw, X_test_raw, y_train, y_test = train_test_split(
#         df1['Review Headline'], df1['Sentiment'], test_size=0.2)
#     vectorizer = TfidfVectorizer()
#     X_train = vectorizer.fit_transform(X_train_raw)
#     X_test = vectorizer.transform(X_test_raw)

#     userString = userStr
#     d = {'a': userString}
#     ser = pd.Series(data=d, index=['a'])
#     string_test = vectorizer.transform(ser)
#     print(string_test)
#     classifierSVC = LinearSVC()
#     filenameofSVC = 'finalized_modelSVC.sav'
#     # Fitting requires training TF_IDF vectors and labels
#     classifierSVC.fit(X_train, y_train)
#     output = classifierSVC.predict(string_test)
#     score = classifierSVC.score(X_test, y_test)
#     print(score)
#     if (output[0] == 1):
#         outputStr = 'positive'
#     elif (output[0] == 0):
#         outputStr = 'neutral'
#     else:
#         outputStr = 'negative'
#     return outputStr
    
def homePage(request):
    return render(request, "ml/index.html")

def showFormData(request):
    outputStr = ""
    prediction = ''
    emoji = ''
    if request.method == 'POST':
        fm = UserInput(request.POST)
        print(fm)
        formData = fm.cleaned_data
        text = formData['write_Review']
        print(formData['write_Review'])
        vector = MachineLearningConfig.vectorizer.transform([text])
        prediction = MachineLearningConfig.model.predict(vector)[0]
        if (prediction == 1):
            outputStr = 'positive'
            emoji = '&#128516'
        elif (prediction == 0):
            outputStr = 'neutral'
            emoji = '&#128528'
        else:
            outputStr = 'negative'
            emoji = '&#128532'
    else:
        fm = UserInput()
    return {'form': fm, 'output': outputStr, 'emoji': emoji}
    # return render(request, 'ml/userInput.html', {'form': fm, 'output': outputStr})

def indexBlog(request):
    return render(request, "blogs/index.html")


def productsBlog(request):
    return render(request, "blogs/products.html")

def knifeItem(request):
    return render(request, "ml/userInput.html", showFormData(request))

def tableItem(request):
    return render(request, "ml/table.html", showFormData(request))

def chairItem(request):
    return render(request, "ml/chair.html", showFormData(request))

# result('haba')

    # lis = []
    #     # lis.append(request.GET['Pregnancies'])
    # print(lis)
