#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import spacy

from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load the dataset
tweets=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment11\Elon_musk.csv",encoding='Latin-1')
tweets.drop(['Unnamed: 0'],inplace=True,axis=1)
tweets


# ## Text Preprocessing

# In[3]:


tweets=[Text.strip() for Text in tweets.Text] # remove both the leading and the trailing characters
tweets=[Text for Text in tweets if Text] # removes empty strings, because they are considered in Python as False
tweets[0:10]


# In[4]:


# Joining the list into one string/text
tweets_text=' '.join(tweets)
tweets_text


# In[5]:


# remove Twitter username handles from a given twitter text. (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
tweets_tokens=tknzr.tokenize(tweets_text)
print(tweets_tokens)


# In[6]:


# Again Joining the list into one string/text
tweets_tokens_text=' '.join(tweets_tokens)
tweets_tokens_text


# In[7]:


# Remove Punctuations 
no_punc_text=tweets_tokens_text.translate(str.maketrans('','',string.punctuation))
no_punc_text


# In[8]:


# remove https or url within text
import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text


# In[9]:


from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)
print(text_tokens)


# In[10]:


# Tokenization
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[11]:


# Tokens count
len(text_tokens)


# In[12]:


# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[13]:


# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[14]:


# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[15]:


# Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[16]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[17]:


clean_tweets=' '.join(lemmas)
clean_tweets


# ## Feature Extaction

# ### 1. Using CountVectorizer

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)


# In[19]:


print(cv.vocabulary_)


# In[20]:


print(cv.get_feature_names()[100:200])


# In[21]:


print(tweetscv.toarray()[100:200])


# In[22]:


print(tweetscv.toarray().shape)


# ### 2. CountVectorizer with N-grams (Bigrams & Trigrams)

# In[23]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[24]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# ### 3. TF-IDF Vectorizer

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[26]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())


# ### Generate Word Cloud

# In[27]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# ## Named Entity Recognition (NER)

# In[28]:


# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[29]:


for token in doc_block[100:200]:
    print(token,token.pos_)    


# In[30]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[31]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[32]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# ## Emotion Mining - Sentiment Analysis

# In[33]:


from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(tweets))
sentences


# In[34]:


sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df


# In[35]:


# Emotion Lexicon - Affin
affin=pd.read_csv("C:\Users\sagar\Desktop\sagar\affiin.csv",sep=',',encoding='Latin-1')
affin


# In[36]:


affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores


# In[37]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[38]:


# manual testing
calculate_sentiment(text='great')


# In[39]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[40]:


# how many words are there in a sentence?
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']


# In[41]:


sent_df.sort_values(by='sentiment_value')


# In[42]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[43]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[44]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[45]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[46]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[47]:


# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


# In[48]:


# Correlation analysis
sent_df.plot.scatter(x='word_count',y='sentiment_value',figsize=(8,8),title='Sentence sentiment value to sentence word count')


# In[ ]:




