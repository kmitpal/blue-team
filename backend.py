import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


#message or news
def news_message(text):
# Load the dataset
    df = pd.read_csv('news_message.csv')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['title'], df['category'], test_size=0.2, random_state=42)

    # Convert the text data into numerical features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier on the training set
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict the labels for the testing set
    y_pred = classifier.predict(X_test)

    # Calculate the accuracy of the classifier on the testing set
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # Use the classifier to predict the label of a new text
    X_new = vectorizer.transform([text])
    prediction = classifier.predict(X_new)[0] 
    # print(prediction)
    if prediction == "news" :
        # st.write('This is a news article')   
        testing(text)
    elif prediction == "message":
        global result
        result = 'Irrelevant'




# real or fake

def clean_word(text):
  text = text.lower()
  text = re.sub('\[.*?\]',' ',text)
  text = re.sub('\\W',' ',text)   #removal of special characters and numbers
  text = re.sub('https?://\$+|www\.\S+'," ",text)    
  text = re.sub('<.*?>+',' ',text)    # removal of html tags
  text = re.sub('[%s]'%re.escape(string.punctuation),' ',text)    #removes punctuation
  text = re.sub('\n',' ',text)
  text = re.sub('\w*\d\w*',' ',text)
  return text

#manual input
clf = joblib.load('nb_model.joblib')
lr=joblib.load('lr_model.joblib')
vector = joblib.load('vectorizer.joblib')
def output(n):
  if(n==0):
    return "Fake News"
  elif (n==1):
    return "Real News"
def testing(news):
  test = {"text":[news]}
  new_test = pd.DataFrame(test)
  new_test['text'] = new_test['text'].apply(clean_word)
  new_x = new_test['text']
  new_xv = vector.transform(new_x)
  pred_lr = lr.predict(new_xv)
  pred_nb = clf.predict(new_xv)
  global result 
  result = output(pred_lr[0]) and output(pred_nb[0])



  # url not present
def extract_keywords(text):
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the text
    X = vectorizer.fit_transform([text])

    # Get the feature names and their corresponding scores
    feature_names = vectorizer.get_feature_names_out()
    feature_scores = X.toarray()[0]

    # Sort the features by their scores and get the top 5
    sorted_scores = sorted(zip(feature_names, feature_scores), key=lambda x: x[1], reverse=True)
    top_keywords = [x[0] for x in sorted_scores[:5]]

    return top_keywords
def check_keyword_similarity(sentences):
    # Extract important keywords from the first sentence
    keywords = set(extract_keywords(sentences[0]))

    # Check if the keywords are present in the other three sentences
    for sentence in sentences[1:]:
        if not set(extract_keywords(sentence)).intersection(keywords):
            return False
    return True



def scraper(Title):
    # Get the user's search query
    # query = input('Enter a search query: ')

    # Construct the URL for the Google News search
    url = 'https://news.google.com/search'
    params = {'q': Title, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'}
    response = requests.get(url, params=params)

    # Parse the HTML content of the search results page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the first 3 headlines and URLs of the top news articles
    articles = soup.find_all('a', class_='DY5T1d')
    count = 0
    i=0
    list1=[]
    for article in articles:
        title = article.text
        url = 'https://news.google.com' + article['href'][1:]
        # print(f'{title}\n')
        list1.append(title)
        count += 1
        if count == 3:
            break
    check_keyword_similarity(list1)




def insta_scrapper(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    meta_tag = soup.find('meta', attrs={'property': 'og:description'})
    content = meta_tag.get('content')
    username = content.split('@')[-1].split()[0]
    # return username 
    global qwerty 
    qwerty = username
    print(qwerty)

    df = pd.read_csv("trusted_sources.csv")
    instagram_acc=df["Instagram"].tolist()
    if qwerty in  instagram_acc:
       print("true news")      
    else:
        scraper(Title)
        # print("fake")



def facebook_scraper(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    page_name = soup.find('title').text.split(' | ')[0]
    # return page_name
    global qwerty1
    qwerty1 = page_name
    q = qwerty1.replace(' - Home','')
    print(q)

    
    df = pd.read_csv("trusted_sources.csv")
    facebook_acc=df["Facebook"].tolist()
    if q in facebook_acc:
       print("true")
    else:
        scraper(Title)
        # print("fake")


# Define a function to write data to CSV
def write_to_csv(data):
    # Load the existing CSV data
    df = pd.read_csv('file.csv')

    # Append the new data to the DataFrame
    df = df.append(data, ignore_index=True)

    # Write the updated DataFrame to CSV
    df.to_csv('file.csv', index=False)

# Create a form for user input
st.set_page_config(page_title="News Detector",page_icon="🔍")
page_bg_img = """
<style>
[data-testid = "stAppViewContainer"]{
background: linear-gradient(#292a5c,#292a5c);
background-size:cover;
}
[data-testid="stHeader"]{
background-color:rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("FAKE NEWS DETECTOR")
with st.form("my_form"):
    # Add form inputs
    Title = st.text_input("Enter Title",placeholder="Enter title")
    Text = st.text_area("Enter Text", placeholder="Enter text")
    url = st.text_input("Enter URL",placeholder="Enter url")

    # Add a submit button
    submit_button = st.form_submit_button(label='SUBMIT')

   

# When the form is submitted, write the data to CSV
if submit_button:
    if "instagram.com" in url:
       user = insta_scrapper(url)
    elif "facebook.com" in url:
        user = facebook_scraper(url)
    elif "" in url:
       st.write("")
       user = "unknown source"    
    data = {'TITLE': Title, 'TEXT': Text, 'URL': url, 'SOURCE': user}
    write_to_csv(data)  
    # Call your function and display the result
    news_message(Title)
    st.write(f"Result: {result}")  


# from pyngrok import ngrok
 
# public_url = ngrok.connect('8501')
# public_url

