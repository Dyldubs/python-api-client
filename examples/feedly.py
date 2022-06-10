print('Loading general libraries...\n')
from bs4 import BeautifulSoup
import csv
from datetime import datetime
from dotenv import load_dotenv
import html
import json
import os
import requests
import time
import urllib.parse
import pandas as pd

#FUTURE PREDICTIONS
#GENERATIVE KEYTERMS

# Function to return current timestamp
def get_ms_now():
	ms = int(time.time()) * 1000
	return ms


# Millisecond timestamp to date function
def ms_to_dt(timestamp):
	ms = timestamp / 1000
	dt = datetime.fromtimestamp(ms).strftime('%Y-%m-%dT%H:%M:%SZ')
	return dt


# Check json for a key:value pair and return value
def check_json_key(json_key, json_object):
	if json_key in json_object:
		return json_object[json_key]
	else:
		return '-'


# Get current timestamp less nominal 24 hours
newer_than = get_ms_now() - (86400 * 1000)

# Instantiate BART
print('Loading our BART model for the summaries...\n')
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

#open output.csv, truncate, then write columns
f = open('../output.csv', "w")
f.truncate()
f.write("Datetime,Title,URL,Author,Section,Publication,Keywords,Summary,WMD_Distance,Jaccard Index,Terms\n")
f.close()

# Load API key
load_dotenv()
key = os.environ.get("key")

class Auth:
    """
    simple class to manage tokens
    """

    def __init__(self, client_id: str = "feedlydev", client_secret: str = "feedlydev"):
        self.client_id: str = client_id
        self.client_secret: str = client_secret
        self._auth_token: str = "A01koTVC9od2NHtRgI0DvohfxoQXC0QepFqSQR4DM6hFi8W4FUTKYIgp5bCvxHUJ2O7IpgV29zwxGoPii6G3zEYf9Vtq_ClBAZvTJmWTPXmT5B9-qlUA8nxMFFbScy7aDXI57YoDwHuFtmQLGqchXkEKZGVNO1Qw_HDiC29SA98YITisDoY0JN80MpQkpnXw9lnxALCLZX_m3kSWdC92srmYYy9a5T5Hhx7UgzSlkWXGFkL2Eu-aSsYDvhI5Rw:feedlydev"
        self.refresh_token: str = None

    @property
    def auth_token(self):
        return self._auth_token

    @auth_token.setter
    def auth_token(self, token: str):
        self._auth_token = token

key = Auth()
key = key.auth_token
# Custom request header
headers = {
	'Authorization': key
}

# Base url
base_url = 'https://cloud.feedly.com/v3/'

#filter terms
terms = ['agile', 'analytics', 'artificial intelligence', 'automation', 'climate', 'cybersecurity', 'demographic', 'digital',
'disruption', 'future', 'privacy', 'security', 'superannuation', 'taxpayer', 'technology', 'trends', 'trust']

# Send request to collections api and access data
url = base_url + 'collections'
response = requests.get(url, headers = headers)


# If request was successful, continue with the script
if response.ok:
	# Capture data
	results = json.loads(response.text)
else:
	print('Something went wrong, check the collections request.')
	exit()

arr = []


# Create a list of 'followed' feeds
collections = []
for result in range(len(results)):
	for feed in results[result]['feeds']:
		collections.append(feed['id'])
		print('{} has been added to your collection...\n'.format(feed['id']))

# Iterate through collections list
for collection in collections:

	# Encode feed id
	feed_id_raw = collection
	feed_id_enc = urllib.parse.quote(feed_id_raw, safe='')


	# Send request to streams api and access data
	url = base_url + 'streams/' + feed_id_enc + '/contents?count=1000&newerThan=' + str(newer_than)
	response = requests.get(url, headers = headers)


	# If request was successful, continue with the script
	if response.ok:
		# Capture data
		results = json.loads(response.text)
	else:
		print('Something went wrong, check the streams request.')
		exit()


	# Iterate through streams data
	for result in range(len(results['items'])):
		
		# Get article count
		article_count = result + 1
		print(article_count)

		# Check if date exists
		date = ms_to_dt(check_json_key('published', results['items'][result]))

		# Check if title exists
		title = check_json_key('title', results['items'][result])

		# Check if url exists
		url = check_json_key('alternate', results['items'][result])[0]['href']

		# Check if author exists
		author = check_json_key('author', results['items'][result])
		
		# Check if section exists
		section = check_json_key('title', results)

		# Check if publication exists
		publication = check_json_key('id', results)

		# Check if keywords exists
		keywords = check_json_key('keywords', results['items'][result])

		# Check if summary exists then check if content exists
		summary_raw = check_json_key('summary', results['items'][result])
		content = check_json_key('content', summary_raw)
		summary = BeautifulSoup(html.unescape(content), features="html.parser").text

		# Bart abstract text summarization
		if len(summary) > 2500:
			inputs = tokenizer.batch_encode_plus([summary], return_tensors='pt', max_length=100, truncation=True)
			summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
			summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 

		row = [date, title, url, author, section, publication, keywords, summary, 0, 0, 0, 0, 0, 0]
		arr.append(row)	
		
		#if any(n in summary for n in terms) or any(n in keywords for n in terms): 
		#	print('Date: {}, Title: {}, URL: {}, Author: {}, Section: {}, Publication: {}, Keywords: {}, Summary: {}\n'.format(date, title, url, author, section, publication, keywords, summary))

		# Write row to csv
		#	with open('../output.csv', 'a', encoding='utf-8-sig') as f:
		#		writer = csv.writer(f)
		#		writer.writerow(row)

#Gensim Word Mover's Distance
sentence_keywords = 'agile analytics artificial intelligence automation climate cybersecurity demographic digital disruption future privacy security superannuation taxpayer technology trends trust'
from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
model = api.load('word2vec-google-news-300')
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

#remove stopwords from our sentences
def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

sentence_keywords = preprocess(sentence_keywords)

#compute WMD using the ``wmdistance`` method.
#In simple terms, we are turning each word into an object.
#Each of these objects can be compared, and how similar
#each of the objects are represents how similar those words
#are semantically
for row in arr:
	row_sentence = preprocess(row[1].lower() + " " + " ".join(row[6]).lower() + " " + row[7].lower())
	distance = model.wmdistance(sentence_keywords, row_sentence)
	distance = float("{0:.4f}".format(distance))
	print('distance = %.4f' % distance)
	row[8] = distance
	#Jaccard Index
	#In simple terms, this calculates the the intersection of
	#the two corpora, with the union, the find the percentage of
	#words in the text that match
	set1 = set(row_sentence)
	set2 = set(terms)
	row[9] = len(set1.intersection(set2)) / len(set1.union(set2))


#Naive keyterm matching
#iterate through terms, note if present in title/keywords/summary field
for keyword in terms:
	for row in arr:
		if keyword in row[1].lower() or keyword in row[7].lower() or keyword in [x.lower() for x in row[6]]:#[x.lower() for x in row[7]]:
			row[10] = row[10] + 1

#sort array by amount of unique terms present in title/keywords/summary field
arr = sorted(arr, key=lambda x:x[8])

# Create an empty dataframe to append list of article summaries
df = pd.DataFrame(arr, columns=["date", "title", "url", "author", "section", "publication", "keywords", "Summary", "wm_distance", "Jaccard_index", "Keyterms", "Subjectivity", "Polarity", "Analysis"])
# Show the first 5 rows of data
print(df.head())

print('Downloading packages for sentiment analysis...\n')
import re
import math
from collections import defaultdict, Counter
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
try:
    nltk.data.find('tokenizers/punkt', 'wordnet')
except LookupError:
    nltk.download(['punkt', 'wordnet', 'omw-1.4'], quiet=True)
from nltk import ngrams
plt.style.use('fivethirtyeight')
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

def removeApostrophe(summary):
    '''
    Remove the apostrophe from text
    '''
    phrase = re.sub(r"won't", "will not", summary)
    phrase = re.sub(r"can't", "can not", summary)
    phrase = re.sub(r"n\'t", " not", summary)
    phrase = re.sub(r"\'re", " are", summary)
    phrase = re.sub(r"\'s", " is", summary)
    phrase = re.sub(r"\'d", " would", summary)
    phrase = re.sub(r"\'ll", " will", summary)
    phrase = re.sub(r"\'t", " not", summary)
    phrase = re.sub(r"\'ve", " have", summary)
    phrase = re.sub(r"\'m", " am", summary)
    return phrase

def removeHTMLtags(text):
    text = BeautifulSoup(text, 'html.parser')
    return text.get_text()

def removeSpecialChars(summary):
    return re.sub('[^a-zA-Z]', ' ', summary)

def removeAlphaNumericWords(summary):
    return re.sub("\S*\d\S*", "", summary).strip()

def cleanText(summary):
    '''
    Performs tokenization, removal of stopwords, lowercasing and lemmatization
    '''
    summary = removeHTMLtags(summary)
    summary = removeApostrophe(summary)
    summary = removeAlphaNumericWords(summary)
    summary = removeSpecialChars(summary)

    summary = summary.lower() # Lower casing
    summary = summary.split() # Tokenization

    # Removing stopwords and lemmatization
    lmtzr = WordNetLemmatizer()
    summary = [lmtzr.lemmatize(word, 'v') for word in summary if not word in set(stopwords.words('english'))]

    summary = " ".join(summary)
    return summary

# Create a corpus for n-gram analysis and word frequency
print("\n It's time to clean the text! \n")   
corpus = []
for index, row in tqdm(list(df.iterrows()), desc='Overall Progress'):
    summary = cleanText(row['Summary'])
    corpus.append(summary)

# Create a list for sentiment analysis
result = []
for art in df['Summary']:
    text = BeautifulSoup(art, 'html.parser') # Remove HTML tags
    text = text.get_text()
    text = "".join(line.strip() for line in text.split("\n")) # Remove whitespaces
    text = re.sub(r'https?:\/\/\S+', '', text) # Remove the hyperlink
    text = re.sub(r'#', '', text) # Removing the '#' symbol
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # Remove @mentions
    result.append(text)
print('\n Finished cleaning the text! \n')    

# Show the cleaned text and add key terms
df['Summary'] = result 
df['Key Terms'] = corpus
print(df.head())

# Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create two new columns
df['Subjectivity'] = df['Summary'].apply(getSubjectivity)
df['Polarity'] = df['Summary'].apply(getPolarity)
print('\n Calculating subjectivity and polarity scores... \n')

# Show the new dataframe with the new columns
print(df.head())

# Plot The Word Cloud
allWords = ' '.join( [art for art in df['Summary']] )
wordCloud = WordCloud(width=500, height=300, background_color = 'white', stopwords = stopwords.words('english'), random_state=21, max_font_size=110).generate(allWords)
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Create a function to compute the negative, neutral, and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)
print('\n Undertaking sentiment analysis... \n')

# Show the dataframe and sentiment examples
print(df.head())

print('\n ----- Examples of positive articles ----- \n')
pprint(list(df[df['Analysis'] == 'Positive'].Summary)[:3], width=400)

print('\n ----- Examples of negative articles ----- \n')
pprint(list(df[df['Analysis'] == 'Negative'].Summary)[:3], width=400)

# Show the sentiment analysis results for the artice category
print('\nNumber (#) of articles that are:\n', df.Analysis.value_counts())
print('\nPecentage (%) of articles that are:\n', df.Analysis.value_counts(normalize=True).mul(100).round(1).astype(str)+ '%')

# Plot the polarity and subjectivity
plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Blue')
    plt.annotate(i,(df['Polarity'][i], df['Subjectivity'][i]), xytext=(math.sqrt(i)/2+2, 0), textcoords='offset points')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# Tokenize the summary of the article into a workable word list
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Key Terms']), axis=1)
print('\n')
print(df.head())

# Find the top 5 keywords
df['Top 5 Word Counts'] = df.apply(lambda row: nltk.FreqDist(row['tokenized_sents']).most_common(5), axis=1)
print('\n')
print(df.head())

# Compile tokenized sentences from all articles to find the most common words
word_tokens = []
for sent in df['tokenized_sents']:
    for word in sent:
        word_tokens.append(word)

# Remove the stopwords
stop_words = set(stopwords.words('english'))
filtered_sentence = []

for w in word_tokens:
    if w.lower() not in stop_words:
        filtered_sentence.append(w)

# Remove punctuation
new_word_tokens = [word for word in filtered_sentence if word.isalnum()]

# Create a frequency distribution
fd = nltk.FreqDist(new_word_tokens)
print('\n ----- 10 most common words across all articles ----- ')
fd.tabulate(10)

# Find most common collocations
finder = nltk.collocations.TrigramCollocationFinder.from_words(new_word_tokens)
print('\n ----- 5 most common collocations ----- ')
print(finder.ngram_fd.most_common(5), '\n')
 
def tokenize(string):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    return re.findall(r'\w+', string.lower())

def count_ngrams(lines, min_length=2, max_length=4):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """

    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams

def print_most_frequent(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')

# Obtain the top 10 most frequently occurring two, three and four word consecutive combinations
ngrams = count_ngrams(df["Key Terms"])
print_most_frequent(ngrams)

# Fetch only the ranking by key terms
##dump array to .csv
#for row in arr:
#	with open('../output.csv', 'a', encoding='utf-8-sig') as f:
#		writer = csv.writer(f)
#		writer.writerow(row)

# Drop tokenized and lemmatized word lists from dataframe
df.drop(['Key Terms', 'tokenized_sents'], axis=1)
# Fetch the dataframe
df.to_csv('../output.csv')