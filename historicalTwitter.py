# Import packages.
import urllib3
from bs4 import BeautifulSoup
import pandas
import shutil
import requests
import warnings
from selenium import webdriver
import time
from itertools import chain, combinations
import cv2
from keras.preprocessing.image import img_to_array
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import os
import re
import matplotlib.image as mpimg

def build_X(path):
    
    X = []
    imgArray = []
    size = (200,200)

    for file in os.listdir(path):
        img = cv2.imread(path+file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size)
        img = img/255
        img = img_to_array(img)
        img = np.squeeze(img,axis=2)
        img = np.hstack(img)
        X.append(img)
        imgArray.append(file)
        
    return X, imgArray

def scrapTweets(sub_concept_sets,browser,image_path,start_date,end_date):
    totaalimgurls = []
    for concept_set_iter in sub_concept_sets:
        
        # Initialize url.
        url = "https://twitter.com/search?l=nl&q="
        
        # loop over concept set entries.
        for concept in concept_set_iter:
            
            # Append concept to url.
            url = url + "%20" + concept
            
        # Append start date to url.
        url = url + "%20since%3A" + start_date
        
        # Append end date to url.
        url = url + "%20until%3A" + end_date
        
        # Append final 
        # Get url.
        browser.get(url)
        
        lenOfPage = browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;") 
        match=False
        while(match==False):
            lastCount = lenOfPage
            time.sleep(2)
            lenOfPage = browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            if lastCount==lenOfPage:
                match=True
                
        source_data = browser.page_source
        soup = BeautifulSoup(source_data,"lxml")
        tweets = soup.find_all('li','js-stream-item')
        
        tweet_data = pandas.DataFrame()
        counter = 0

        imgurls = []
        
        for tweet in tweets:
            
            if tweet.find('p','tweet-text'):
                tweet_user = tweet.find('span','username').text
                tweet_text = tweet.find('p','tweet-text').text.encode('utf8')
                tweet_id = tweet['data-item-id']
                tweet_img = tweet.find('div',{"class":"AdaptiveMedia-photoContainer js-adaptive-photo "})
                
                if (str(type(tweet_img)) != "<class 'NoneType'>"):
                    tweet_img = tweet_img['data-image-url']
                    imgurls.append(tweet_img)
                    response = requests.get(tweet_img, stream=True)
                    with open(image_path + 'img_' + tweet_id + '.png', 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                    del response
                                
                timestamp = tweet.find('a','tweet-timestamp')['title']
                tweet_total = [tweet_user,tweet_text,tweet_id,timestamp]
                tweet_data[counter] = tweet_total
                    
                counter += 1
            else:
                continue
        
        tweet_data = tweet_data.transpose()
        
        if (len(tweet_data) > 0):
            tweet_data.columns = ["user_name","text","id","timestamp"]
        totaalimgurls.append(imgurls)
    return tweet_data, totaalimgurls


def clean(doc,stop,exclude,lemma):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def display_topics(model, feature_names, no_top_words):
    topicList = []
    for topic_idx, topic in enumerate(model.components_):
        topic  = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        topicList.append(topic)
    return topicList

def computeTopic(data,n_clus,no_top_words):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    nmf = NMF(n_components=n_clus, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    
    return display_topics(nmf, tfidf_feature_names, no_top_words)

def retrievelInfo(tag):
# Disable warnings.
    warnings.filterwarnings("ignore")   

    # Define concept set.
    concept_set = [tag]

    # Determine subsets of concept set.
    sub_concept_sets = list(chain(*map(lambda x: combinations(concept_set, x), range(1, len(concept_set)+1))))

    # Define start date and end date.
    start_date = "2015-01-01"
    end_date = "2015-02-01"

    # Construct web browser.
    http = urllib3.PoolManager()
    browser = webdriver.Chrome('/Users/bart/Documents/OdysseyHack/chromedriver')

    # Define image folder name.
    image_folder_name = "stormschade"

    # Define image folder path.
    image_path = '/Users/bart/Documents/OdysseyHack/Data/' + image_folder_name + '/'

    # Create image folder.
    if os.path.exists(image_path):
        shutil.rmtree(image_path)
    os.makedirs(image_path)

    # For topic modeling 
    no_top_words = 5

    # Loop over sub concept sets.
    tweet_data, imgurls =  scrapTweets(sub_concept_sets,browser,image_path,start_date,end_date)

    # Close webdriver.
    browser.close()

    # Translate images to numpy array.
    X, imgArray = build_X(image_path)

    # Embed images in two dimensional space.
    X_embedded = TSNE(n_components=2).fit_transform(X)

    # Cluster images with k-means.
    n_clus = 4
    kmeans = KMeans(n_clusters=n_clus)
    kmeans.fit(X_embedded)
    y_kmeans = kmeans.predict(X_embedded)

    # Plot clustered images.
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50, alpha=0.5);
    # plt.show()

    # Loop over images.
    for i in range(len(imgArray)):
        
        # Extract k-means label.
        k_means_label = y_kmeans[i]
        
        # Define label directory path.
        label_path = image_path + "label_" + str(k_means_label) + "/"
        
        # Create label storage folder.
        if not os.path.exists(label_path):
            os.makedirs(label_path)
            
        # Store image in label folder.
        shutil.move(image_path + imgArray[i],label_path + imgArray[i])
        
    data = tweet_data['text'].tolist()

    for i in range(len(data)):

        data[i] = str(data[i])
        data[i] = re.sub(r'http\S+', '', data[i])
        data[i] = re.sub(r"pic\.twitter\.com\/\w*", "", data[i])
        data[i] = re.sub(r"[^@]+@[^@]+\.[^@]+", "", data[i])  # remove email addresses
        data[i] = re.sub(r"\@\w+", "", data[i])  # remove user names
        data[i] = re.sub(r'\b\w{1,3}\b', '', data[i])
        
    stop = set(stopwords.words('dutch'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()        
    doc_clean = [clean(doc,stop,exclude,lemma).split() for doc in data]

    textList = []
    # def getTexts():
    #     return textList

    imgList = []
    # def getImages():
    #     return imgList

    # Loop over centers.
    for center in centers:
        
        # Compute Euclidian distance.
        eucl_dis = np.sqrt(np.sum((center-X_embedded) ** 2,axis=1))
        
        # Find index of minimum euclidan distance.
        min_index = np.argmin(eucl_dis)
        
        # Extract central image name.
        central_image_name = imgArray[min_index]
        
        # Update label path.
        label_path = image_path + "label_" + str(y_kmeans[min_index]) + "/"
        
        # Update central image path.
        central_image_path = label_path + central_image_name
        
        # Replace backslash with forward slash.
        # central_image_path = central_image_path.replace("/","\/")
        
        # Read central image.
        imgList.append(imgurls[0][min_index])
        central_image = cv2.imread(central_image_path,cv2.COLOR_BGRA2RGBA)   

        # Show central image.
        # plt.imshow(central_image)
        # plt.show()
        
        # Plot text.
        textList.append(computeTopic(data,n_clus,no_top_words)[y_kmeans[min_index]])

    return imgList,textList
   


