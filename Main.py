import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import re
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
import re
import nltk
import string
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords #imports stopwords from nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
#import spacy
#nlp = spacy.load('en_core_web_sm')
import pandas as pd
import os
from flask import Flask, render_template,request
#from google.colab import files
import requests
from bs4 import BeautifulSoup
import sys
from PIL import Image
import base64
import io


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route("/result", methods = ["GET","POST"])
def result():
    if request.method == "POST":
        # getting input with url = url in HTML form
        url = str(request.form.get("url"))
        print(url)
        header = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'referer': 'https://www.amazon.in/'}

        search_response = requests.get(url, headers=header)

        print(search_response.status_code)

        search_response.text

        cookie = search_response.cookies

        def Searchreviews(review_link):
            url = "https://www.amazon.in" + review_link
            print(url)
            page = requests.get(url, cookies=cookie, headers=header)
            if page.status_code == 200:
                return page
            else:
                return "Error"

        # data_asin=[]
        print(url)
        response = requests.get(url, headers=header)

        response.status_code

        link = []
        soup = BeautifulSoup(response.content, features="lxml")
        for i in soup.findAll("a", {'data-hook': "see-all-reviews-link-foot"}):
            link.append(i['href'])

        print(link)

        reviews = []
        for j in range(len(link)):
            for k in range(50):
                response = Searchreviews(link[j] + '&pageNumber=' + str(k))
                soup = BeautifulSoup(response.content, features="lxml")
                for i in soup.findAll("span", {'data-hook': "review-body"}):
                    reviews.append(i.text)

        if (len(reviews)) == 0 :
            print('Error 0 reviews found')
            return render_template("index.html", prediction_text='Reviews could not be scrapped :/  Only Amazon links work !')


        rev = {'reviews': reviews}

        review_data = pd.DataFrame.from_dict(rev)
        pd.set_option('max_colwidth', 800)

        print(review_data.head(5))

        print(review_data.shape)

        review_data.to_csv('Bye.csv')

        print('File saved... !')

        def datapre(sen):
            sen = re.sub(r"didn't", "did not", sen)
            sen = re.sub(r"don't", "do not", sen)
            sen = re.sub(r"won't", "will not", sen)
            sen = re.sub(r"can't", "can not", sen)
            sen = re.sub(r"wasn't", "was not", sen)
            sen = re.sub(r"\'ive", " i have", sen)
            sen = re.sub(r"\'m", " am", sen)
            sen = re.sub(r"\'ll", " will", sen)
            sen = re.sub(r"\'re", " are", sen)
            sen = re.sub(r"\'s", " is", sen)
            sen = re.sub(r"\'d", " would", sen)
            sen = re.sub(r"\'t", " not", sen)
            sen = re.sub(r"\'m", " am", sen)
            return sen

        def clean_Review(review_text):
            review_text = re.sub(r'http\S+', '', review_text)
            review_text = re.sub('[^a-zA-Z]', ' ', review_text)
            review_text = str(review_text).lower()
            review_text = word_tokenize(review_text)

            review_test = [word for word in review_text if word not in stopwords.words('english')]

            review_text = [item for item in review_text if item not in stop_words]
            review_text = [lemma.lemmatize(word=w, pos='v') for w in review_text]
            review_text = [i for i in review_text if len(i) > 3]
            review_text = ' '.join(review_text)
            return review_text

        model = pickle.load(open('trained_model.pkl', 'rb'))
        fittrans = pickle.load(open('fittrans.pkl', 'rb'))
        vocabb = pickle.load(open('vocab.pkl', 'rb'))
        vecccc = pickle.load(open('vec_s.pkl', 'rb'))

        test_data = pd.read_csv("Bye.csv")
        test_data['reviews'].head(10)
        clean = []
        for sent in test_data['reviews'].values:
            fa = clean_Review(sent)
            da = datapre(fa)
            clean.append(da)
        test_data['cleane'] = clean
        test_data.to_csv('man.csv')
        test_data['cleane'].shape
        test_data.head()
        vec_s = CountVectorizer()
        # y_pred_M = model.predict(fittrans.transform([test_data.cleane]))
        test_data_features = vecccc.transform(test_data.cleane)

        y_pred_M = model.predict(test_data_features)
        print(y_pred_M)

        y_pred_M.shape

        output = pd.DataFrame(data={"reviews": test_data["reviews"], "sentiment": y_pred_M})
        # Use pandas to write the comma-separated output file
        output.head()
        output.to_csv("Bag_of_Wordsmodel.csv", index=False, quoting=0)
        output.head(10)

        print(output['sentiment'].value_counts())
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        mylabels = ['postive', 'negative']
        plt.pie(output['sentiment'].value_counts(), labels=mylabels, autopct='%1.1f%%')
         #plt.show
        plt.savefig('graph.png')
        plt.clf()
        im = Image.open("graph.PNG")
        data = io.BytesIO()
        im.save(data, "PNG")
        encoded_img_data = base64.b64encode(data.getvalue())

        return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))

       # return render_template("index.html", prediction_text='Reviews scrapped and saved at   {} '.format(1))



        #return url




if __name__ == "__main__":
    app.run(debug=True)















