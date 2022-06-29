import re
import tweepy
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, redirect, url_for, request

api2 = None

app = Flask(__name__ , template_folder='templates')
app.static_folder = 'static'

def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '',text)   # For Removing mentions
    text = re.sub(r'#','',text) # For Removing hashtags
    text = re.sub(r'RT[\s]+','',text) #Remove Re-Tweets
    text = re.sub(r'https?:\/\/\S+','',text) # Remove Hyperlinks
    return text

# Function To get Subjectivity of the tweet
def getSubjectivity(text):
    return round(TextBlob(text).sentiment.subjectivity,3)

# Function to get Polarity of the tweet
def getPolarity(text):
    return round(TextBlob(text).sentiment.polarity,3)

def analysis(score):
    if(score > 0):
        return  'Positive'
    elif(score < 0):
        return 'Negative'
    else:
        return 'Neutral'


def get_user_tweets(api,user_name,count=5):
    count = int(count)

    posts = tweepy.Cursor(api.user_timeline, screen_name=user_name,lang ='en', count=5,       tweet_mode="extended" ).items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

    df['Tweets'] = df['Tweets'].apply(cleanText)
    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
    df['Polarity'] = df['Tweets'].apply(getPolarity)
    df['Sentiment'] = df['Polarity'].apply(analysis)

    return df


def get_hashtag_tweets(api,hashtag,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.search_tweets, q=hashtag, count=100,lang ='en', tweet_mode="extended").items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

    df['Tweets'] = df['Tweets'].apply(cleanText)
    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
    df['Polarity'] = df['Tweets'].apply(getPolarity)
    df['Sentiment'] = df['Polarity'].apply(analysis)

    return df

def plot(df,name):
    plt.title('Sentiment Analysis Result of '+name)
    plt.ylabel('Counts')
    df['Sentiment'].value_counts().plot(kind='bar')
    plt.xticks(rotation=0)
    plt.savefig('static/my_plot.png')
    plt.switch_backend('agg')
    # plt.show()

@app.route('/')
def home():
  return render_template("index.html")


@app.route("/predict_user", methods=['POST','GET'])
def predict_user():
    if request.method == 'POST':
        user_name = request.form['user_name']
        count = request.form['count']
        fetched_tweets = get_user_tweets(api2, user_name, count)
        plot(fetched_tweets,user_name)
        fetched_tweets = fetched_tweets.to_dict('records')

        return render_template('result_user.html', result=fetched_tweets)


@app.route("/predict_tag", methods=['POST','GET'])
def predict_tag():
    if request.method == 'POST':
        hashtag = request.form['hashtag']
        count = request.form['count']
        fetched_tweets = get_hashtag_tweets(api2,hashtag,count)
        plot(fetched_tweets,hashtag)

        fetched_tweets = fetched_tweets.to_dict('records')
        return render_template('result_user.html', result=fetched_tweets)

@app.route("/predict_sentence",methods=['POST','GET'])
def predict_text():
    if request.method== 'POST':
        sentence = request.form['sentence']
        Subjectivity = getSubjectivity(sentence)
        Polarity = getPolarity(sentence)
        Sentiment = analysis(Polarity)

        return render_template('result_sentence.html',sentence=sentence,Subjectivity=Subjectivity,Polarity=Polarity,Sentiment=Sentiment)

if __name__ == '__main__':

    API_Key = "YOfX4r7g3YCy5D2vE3BUG115I"
    API_Key_Secret = "GbTTmWA5JlEyPMJF9WCTMcyPHeNcXjHBREyjSYiX4iIyHwaLlZ"
    Access_Token = "1502904942344179718-or4wUdWtudoY9HX3hjLjJSmQpvCyBx"
    Access_Token_Secret = "AUFrsoATq5EBBNbpB4o3YdaIxppP1ghq3mfqtMZjGp9PC"

    try:
        authenticate = tweepy.OAuthHandler(API_Key, API_Key_Secret)
        authenticate.set_access_token(Access_Token, Access_Token_Secret)
        api = tweepy.API(authenticate, wait_on_rate_limit=True)
        api2 = api
    except:
        print("Error: Authentication Failed")

    app.debug = True
    app.run()