print('Downloading packages...')
from collections import defaultdict, Counter
import collections # to count unigrams
from pprint import pprint
from tqdm import tqdm
import nltk
try:
    nltk.data.find('tokenizers/punkt', 'wordnet')
except LookupError:
    nltk.download(['punkt', 'wordnet', 'omw-1.4'], quiet=True)

print('Loading general libraries...\n')
from bs4 import BeautifulSoup
from feedly.api_client.session import FeedlySession
from feedly.api_client.stream import StreamOptions
from datetime import datetime
import math
import html
import time
import csv
import json
import sys
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
plt.style.use('fivethirtyeight')

# Setting up and authenticating feedly client
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus

from requests import Response, Session
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError

from feedly.api_client.data import FeedlyUser
from feedly.api_client.protocol import (
    APIClient,
    BadRequestAPIError,
    RateLimitedAPIError,
    ServerAPIError,
    UnauthorizedAPIError,
)
# Set timer
start_time = time.time()

logger = logging.getLogger(__name__)


class Auth:
    """
    simple class to manage tokens
    """

    def __init__(self, client_id: str = "feedlydev", client_secret: str = "feedlydev"):
        self.client_id: str = client_id
        self.client_secret: str = client_secret
        self._auth_token: str = None
        self.refresh_token: str = None

    @property
    def auth_token(self):
        return self._auth_token

    @auth_token.setter
    def auth_token(self, token: str):
        self._auth_token = token


class FileAuthStore(Auth):
    """
    a file based token storage scheme
    """

    def __init__(
        self,
        token_dir: Path = Path.home() / ".config/feedly",
        client_id: str = "feedlydev",
        client_secret: str = "feedlydev",
    ):
        """

        :param token_dir: the directory to store the tokens
        :param client_id: the client id to use when refreshing the auth token. the default value works for developer tokens.
        :param client_secret: the client secret to use when refreshing the auth token. the default value works for developer tokens.
        """
        super().__init__(client_id, client_secret)
        if not token_dir.is_dir():
            raise ValueError(f"{token_dir.absolute()} does not exist!")

        refresh_path = token_dir / "refresh.token"
        if refresh_path.is_file():
            self.refresh_token = refresh_path.read_text().strip()

        self.auth_token_path: Path = token_dir / "access.token"
        self._auth_token = self.auth_token_path.read_text().strip()

    @Auth.auth_token.setter
    def auth_token(self, token: str):
        self._auth_token = token
        self.auth_token_path.write_text(token)


class FeedlySession(APIClient):
    def __init__(
        self,
        auth: Optional[Union[str, Auth]] = None,
        api_host: str = "https://feedly.com",
        user_id: str = None,
        client_name="feedly.python.client",
    ):
        """
        :param auth: either the access token str to use when making requests or an Auth object to manage tokens. If none
         are passed, it is assumed that the token and refresh token are correcly setup in the `~/.config/feedly`
         directory. You can run setup_auth.py in the examples to get setup.
        :param api_host: the feedly api server host.
        :param user_id: the user id to use when making requests. If not set, a request will be made to determine the user from the auth token.
        :param client_name: the name of your client, set this to something that can identify your app.
        """
        super().__init__()
        if not client_name:
            raise ValueError("you must identify your client!")

        if isinstance(auth, str):
            token: str = auth
            auth = Auth()
            auth.auth_token = token
        elif auth is None:
            auth = FileAuthStore()

        self.auth: Auth = auth
        self.api_host: str = api_host
        self.session = Session()
        self.session.mount(
            "https://feedly.com", HTTPAdapter(max_retries=1)
        )  # as to treat feedly server and connection errors identically
        self.client_name = client_name
        self.timeout: int = 10
        self.max_tries: int = 3

        user_data = {"id": user_id} if user_id else {}
        self._user: FeedlyUser = FeedlyUser(user_data, self)
        self._valid: bool = None
        self._last_token_refresh_attempt: float = 0

    def __repr__(self):
        return f"<feedly client user={self.user.id}>"

    def __str__(self):
        return self.__repr__()

    def close(self) -> None:
        self._valid = None
        if self.session:
            self.session.close()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def user(self) -> FeedlyUser:
        return self._user

    def do_api_request(
        self,
        relative_url: str,
        method: str = None,
        params: Dict[str, Any] = None,
        data: Dict = None,
        timeout: int = None,
        max_tries: int = None,
    ) -> Union[Dict[str, Any], List[Any], None]:
        """
        makes a request to the feedly cloud API (https://developers.feedly.com), and parse the response
        :param relative_url: the url path and query parts, starting with /v3
        :param params: the query parameters.
        :param data: the post data to send (as json).
        :param method: the http method to use, will default to get or post based on the presence of post data
        :param timeout: the timeout interval
        :param max_tries: the number of tries to do before failing
        :return: the request result as parsed from json.
        :rtype: dict or list, based on the API response
        :raises: requests.exceptions.HTTPError on failure. An appropriate subclass may be raised when appropriate,
         (see the ones defined in this module).
        """
        resp = self.make_api_request(
            relative_url=relative_url, method=method, params=params, data=data, timeout=timeout, max_tries=max_tries
        )
        return resp.json() if resp.content is not None and len(resp.content) > 0 else None

    def make_api_request(
        self,
        relative_url: str,
        method: str = None,
        params: Dict[str, Any] = None,
        data: Dict = None,
        timeout: int = None,
        max_tries: int = None,
    ) -> Response:
        """
        makes a request to the feedly cloud API (https://developers.feedly.com), and parse the response
        :param relative_url: the url path and query parts, starting with /v3
        :param params: the query parameters.
        :param data: the post data to send (as json).
        :param method: the http method to use, will default to get or post based on the presence of post data
        :param timeout: the timeout interval
        :param max_tries: the number of tries to do before failing
        :return: the request result as parsed from json.
        :raises: requests.exceptions.HTTPError on failure. An appropriate subclass may be raised when appropriate,
         (see the ones defined in this module).
        """
        if self.timeout is None:
            timeout = self.timeout

        if max_tries is None:
            max_tries = self.max_tries

        if self.auth.auth_token is None:
            raise ValueError("authorization token required!")

        if relative_url[0] != "/":
            relative_url = "/" + relative_url

        if not relative_url.startswith("/v3/"):
            raise ValueError(
                f"invalid endpoint {relative_url} -- must start with /v3/ See https://developers.feedly.com"
            )

        if max_tries < 0 or max_tries > 10:
            raise ValueError("invalid max tries")

        full_url = f"{self.api_host}{relative_url}"
        if "?client=" not in full_url and "&client=" not in full_url:
            full_url += ("&" if "?" in full_url else "?") + "client=" + quote_plus(self.client_name)

        tries = 0
        if method is None:
            method = "get" if data is None else "post"

        if method == "get" and data is not None:
            raise ValueError("post data not allowed for GET requests")

        try:
            if self.rate_limiter.rate_limited:
                raise RateLimitedAPIError(None)
            while True:
                tries += 1
                if self.rate_limiter.rate_limited:
                    until = datetime.datetime.fromtimestamp(self.rate_limiter.until).isoformat()
                    raise ValueError(f"Too many requests. Client is rate limited until {until}")
                headers = {"Authorization": self.auth.auth_token}
                if data:
                    headers["Content-Type"] = "application/json"

                resp = None
                conn_error = None
                try:
                    resp = self.session.request(
                        method, full_url, headers=headers, timeout=timeout, json=data, params=params
                    )
                except OSError as e:
                    conn_error = e

                if resp:
                    self.rate_limiter.update(resp)

                if not conn_error and resp.ok:
                    return resp
                else:
                    if tries == max_tries or (
                        resp is not None and 400 <= resp.status_code <= 500
                    ):  # don't retry bad requests:
                        if conn_error:
                            raise conn_error
                        else:
                            logger.error(resp.json())
                            resp.raise_for_status()
                    logger.warning("Error for %s: %s", relative_url, conn_error if conn_error else resp.text)
                    time.sleep(2 ** (tries - 1))  # 1 second, then 2, 4, 8, etc.
        except HTTPError as e:
            code = e.response.status_code
            if code == 400:
                raise BadRequestAPIError(e)
            elif code == 401:
                if (
                    not relative_url.startswith("/v3/auth")
                    and self.auth.refresh_token
                    and time.time() - self._last_token_refresh_attempt > 86400
                ):
                    try:
                        self._last_token_refresh_attempt = time.time()
                        auth_data = {
                            "refresh_token": self.auth.refresh_token,
                            "grant_type": "refresh_token",
                            "client_id": self.auth.client_id,
                            "client_secret": self.auth.client_secret,
                        }
                        token_data = self.do_api_request("/v3/auth/token", data=auth_data)
                        self.auth.auth_token = token_data["access_token"]
                        return self.do_api_request(
                            relative_url=relative_url, method=method, data=data, timeout=timeout, max_tries=max_tries
                        )
                    except Exception as e2:
                        logger.info("error refreshing access token", exc_info=e2)
                        # fall through to raise auth error
                raise UnauthorizedAPIError(e)
            elif code == 429:
                if not self.rate_limiter.rate_limited:
                    self.rate_limiter.make_rate_limited()
                raise RateLimitedAPIError(e)
            elif code == 500:
                raise ServerAPIError(e)

            raise e

# Function to return current timestamp
def get_ms_now():
    ms = int(time.time()) * 1000
    return ms

# Get current timestamp less nominal 24 hours
newer_than = get_ms_now() - (864000 * 1000)

# Millisecond timestamp to date function
def ms_to_dt(timestamp):
	ms = timestamp / 1000
	dt = datetime.fromtimestamp(ms).strftime('%Y-%m-%dT%H:%M:%SZ')
	return dt

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

def alphanumeric(lst):
    '''
    Removes non-alpha-numeric characters from list
    '''
    for word in lst:
        if word.isalnum():
            word
        else:
            lst.remove(word)
    return lst

# Instantiate a feedly session
session = FeedlySession()

# Print out available categories
names = session.user.user_categories.name2stream.keys()
print(f"Available categories:")
for n in names:
    print(n)

# Prompt for the category name/id to use
user_category_name_or_id = input("\n> Enter one of the above categories: ")

# Fetch the category by its name/id
category = session.user.user_categories.get(user_category_name_or_id)

# Create an empty dataframe to append article summaries
df = pd.DataFrame(columns=['Summary'])

# Create a dataframe with a column called summary
df = pd.DataFrame([article['summary']['content'] for article in category.stream_contents(options=StreamOptions(max_count=50))], columns=['Summary'])

# Show the first 5 rows of data
print(df.head())

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
df['Keyword Counts'] = df.apply(lambda row: nltk.FreqDist(row['tokenized_sents']).most_common(5), axis=1)
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
elapsed_time = time.time() - start_time
print('Completed in {:.03f} seconds.'.format(elapsed_time))