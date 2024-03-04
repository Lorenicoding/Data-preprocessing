---
Title: Demo Blog Post
Date: 2022-01-31 01:12
Category: Progress Report
Tags: Group Super NLP
---

By Group "Five Sigma"



## Text preprocessing 
Having extracted the minutes from the FOMC website, we initiated the text preprocessing phase to standardize the content and eliminate any extraneous elements. This meticulous process aims to enhance the quality of the text, ensuring optimal conditions for subsequent sentiment analysis.

## Step 1: Read data from csv file

Initially, our first step involves **extracting data from the CSV file**.

The code we use is as follows:
```python
# read csv file
import pandas as pd
df = pd.read_csv('FOMC_text.csv')
```
## Step 2: Remove unrelated paragraphs

Upon importing the data, our task is to examine the content structure and **eliminate irrelevant paragraphs**.
In the context of our FOMC example, we identified paragraphs pertaining to voting members, contact methods and issue dates that were irrelevant to our analysis and consequently removed them. It's noteworthy that these paragraphs shared a common starting phrase, prompting us to address this consistent pattern during our data refinement process.

The code we use is as follows:
```python
# Removal of unrelated paragraph
import re
def Removal_paragraph(text):
    text = re.sub(r'Voting for the monetary policy action.*','',text)
    text = re.sub(r'For media inquiries.*','',text)
    text = re.sub(r'Implementation Note issued.*','',text)
    return text
df['text_related'] = df['Text'].apply(lambda x: Removal_paragraph(x))
```

## Step 3: Convert to lowercase

To further enhance uniformity and facilitate consistent analysis, we proceeded to **convert all text entries to lowercase**.

The code we use is as follows:
```python
# Convert to lowercase
df['lowercase_text'] = df['text_clean'].str.lower()
```

## Step 4: Lemmatization & POS Tagging

For optimal efficiency in transforming words into their meaningful base forms, we employ **lemmatization**. To enhance accuracy, we leverage **Part-of-Speech (POS) tagging** to annotate the grammatical categories of each word. This strategic use of POS tagging ensures that lemmatization is executed with improved precision, contributing to more accurate and meaningful results.

The code we use is as follows:
```python
# Lemmatization & POS Tagging
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V": wordnet.VERB,"J": \
               wordnet.ADJ, "R": wordnet.ADV}
def lemmatize_words(text):
    # find pos tags
    pos_text = pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word,\
    wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])
df['lemmatized_text'] = df['lowercase_text'].apply(lambda x: lemmatize_words(x))
```

## Step 5: Remove punctions within the sentence

Following the standardization of the text, our next step involved the removal of punctuation within the sentences. However, we made a deliberate choice to retain essential punctuation marks such as ".", "?", and "!" to preserve the sentence structure. This decision was driven by the necessity to maintain the integrity of the text in sentence format, as we require sentiment scores for each individual sentence.

The code we use is as follows:
```python
# Removal of Punctions within the sentence
def remove_punctuations(text):
    punctuations = "\"#$%&'()*+-/:;<=>@[\]^_`{|}~"
    return text.translate(str.maketrans('', '', punctuations))
df['clean_text_1'] = df['lemmatized_text'].apply(lambda x: remove_punctuations(x))
```

## Step 6: Remove stopwords

Commonly occurring words like articles, prepositions, and conjunctions, known as stopwords, are abundant but contribute minimally to extracting the essence of text. Following the elimination of punctuation within the sentences, we deliberately **excluded stopwords**. This strategic step allows us to concentrate on the more meaningful and content-rich words, thereby enhancing the flow of analysis and optimizing overall efficiency.

The code we use is as follows:
```python
# Removal of stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in STOPWORDS])
df['clean_text_2'] = df['clean_text_1'].apply(lambda x: remove_stopwords(x))
```

## Step 7: Remove numbers and extra spaces

In the conclusive phase of our text preprocessing process, we systematically **eliminate numerical digits and extraneous spaces**. This step aims to enhance the relevance and meaningfulness of the text specifically for sentiment analysis, as numbers and extra spaces hold little significance in this context.

The code we use is as follows:
```python
# Removal of Punctions within the sentence
import re
def remove_spl_chars(text):
    text = re.sub('[\d]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text
df['clean_text_3'] = df['clean_text_2'].apply(lambda x: remove_spl_chars(x))
```

## Data preprocessing results

We will utilize a minute from the FOMC website as an illustrative example to showcase the outcomes. Through this, we anticipate witnessing **a marked enhancement in relevance and clarity**, particularly conducive to more accurate sentiment analysis.

**original text**
![alt text](<Original text.png>)


**Clean text**
![alt text](<Clean text.png>)





