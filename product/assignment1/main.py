#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import pathlib

# Read txt files in data/gutenberg and put into a df
def read_txt_files():
    # Get all txt files in data/gutenberg
    
    ################################################ 
    # Data folder path
    path = pathlib.Path("./data/gutenberg")
    ################################################
    
    txt_files = path.glob("*.txt")

    # Read each txt file and put into a df
    data = []
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Find title, author, and language
            title = None
            author = None
            language = None
            start = 0
            end = len(lines)
            for i, line in enumerate(lines[:100]):
                if "Title: " in line:
                    title = line.split("Title: ")[1].strip()
                if "Author: " in line:
                    author = line.split("Author: ")[1].strip()
                if "Language: " in line:
                    language = line.split("Language: ")[1].strip()
                
                if (line.__contains__("*** START OF THE PROJECT GUTENBERG EBOOK")):
                    start = i
                    break
                    
            for i, line in enumerate(lines[-1000:]):
                if (line.__contains__("*** END OF THE PROJECT GUTENBERG EBOOK")):
                    end = i
                    break
                
                
            content = "".join(lines[start+1:end-1])
            data.append({"title": title, "author": author, "language": language, "content": content})
    df = pd.DataFrame(data)
    df.set_index("title", inplace=True)
    return df

df = read_txt_files()


# In[58]:


from nltk.tokenize import word_tokenize
import nltk.downloader
nltk.download("punkt")

# Tokenization
df["tokens"] = df["content"].apply(word_tokenize)


# In[59]:


from string import punctuation
from nltk.corpus import stopwords
nltk.download("stopwords")
punctuation = set(punctuation)
stopwords = set(stopwords.words("english"))
punctuation.update(["”", "“", "’"]) # Two quote marks that are not in the punctuation set


# In[60]:


# Filter out punctuation and stopwords
# Testing using lower case for comparison.
df["filtered_tokens"] = df["tokens"].apply(lambda tokens: [token for token in tokens if token.lower() not in punctuation and token.lower() not in stopwords])


# In[61]:


from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Stemming
stemmer = PorterStemmer()
df["stemmed_tokens"] = df["filtered_tokens"].apply(lambda doc: [stemmer.stem(token) for token in doc])

# Lemmatization
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
df["lemmatized_tokens"] = df["filtered_tokens"].apply(lambda doc: [lemmatizer.lemmatize(token) for token in doc])


# In[62]:


from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

def findNGrams(documents):
    unigrams = documents.apply(FreqDist).sum()
    bigrams = BigramCollocationFinder.from_documents(documents)
    bigrams.apply_freq_filter(2)
    trigrams = TrigramCollocationFinder.from_documents(documents)
    trigrams.apply_freq_filter(2)
    
    return unigrams, bigrams.ngram_fd, trigrams.ngram_fd



# In[63]:


# Stats after different processing steps
stats = pd.DataFrame(columns=["TotalTokens", "Types", "Unigrams", "Bigrams", "Trigrams"])

def get_stats(tokens):
    TotalTokens = len(tokens)
    types = len(set(tokens))
    ratio = types / TotalTokens
    avg_token_length = sum([len(token) for token in tokens]) / TotalTokens
    return TotalTokens, types, #ratio, avg_token_length

# Stats after different processing steps
stats = pd.DataFrame(columns=["TotalTokens", "Types", "Unigrams", "Bigrams", "Trigrams"])

processing_steps = [("tokens", df["tokens"]), 
                    ("filtered", df["filtered_tokens"]), 
                    ("stemmed", df["stemmed_tokens"]), 
                    ("lemmatized", df["lemmatized_tokens"])]

for step_name, tokens in processing_steps:
    aggTokens = tokens.sum()
    TotalTokens, Types = get_stats(aggTokens)
    Unigrams, Bigrams, Trigrams = findNGrams(tokens)
    
    stats = pd.concat([stats, pd.DataFrame([[TotalTokens, Types, Unigrams, Bigrams, Trigrams]], columns=stats.columns, index=[step_name])])

# Find top unigrams, bigrams, and trigrams. They are stored in a FreqDist object.
stats["Top_Unigrams"] = stats["Unigrams"].apply(lambda x: x.most_common(10))
stats["Top_Bigrams"] = stats["Bigrams"].apply(lambda x: x.most_common(10))
stats["Top_Trigrams"] = stats["Trigrams"].apply(lambda x: x.most_common(10))

stats["Unigram_Count"] = stats["Unigrams"].apply(len) # Principally the same as Types
stats["Bigram_Count"] = stats["Bigrams"].apply(len)
stats["Trigram_Count"] = stats["Trigrams"].apply(len)

# Weight counts by the number of occurrences
stats["Weighted_Unigrams"] = stats["Unigrams"].apply(lambda x: x.N())
stats["Weighted_Bigrams"] = stats["Bigrams"].apply(lambda x: x.N())
stats["Weighted_Trigrams"] = stats["Trigrams"].apply(lambda x: x.N())


# In[64]:

print("\n\n\n")
print(stats.loc[["tokens", "filtered"]]["TotalTokens"])
print([token[0] for token in stats.loc["tokens"]["Top_Unigrams"]])
print([token[0] for token in stats.loc["filtered"]["Top_Unigrams"]])


# In[66]:


import matplotlib.pyplot as plt

# Ignore the first row, which is the tokenized text
stats = stats.iloc[1:] if stats.index[0] == "tokens" else stats

# Histograms of number of unigrams, bigrams, and trigrams by processing step
fig, ax = plt.subplots(3, 2, figsize=(15, 10))

stats["Unigram_Count"].plot(kind="bar", ax=ax[0, 0], title="Unique Unigram Count")
stats["Bigram_Count"].plot(kind="bar", ax=ax[1, 0], title="Unique Bigram Count")
stats["Trigram_Count"].plot(kind="bar", ax=ax[2, 0], title="Unique Trigram Count")
stats["Weighted_Unigrams"].plot(kind="bar", ax=ax[0, 1], title="Total Unigram Count")
stats["Weighted_Bigrams"].plot(kind="bar", ax=ax[1, 1], title="Total Bigram Count")
stats["Weighted_Trigrams"].plot(kind="bar", ax=ax[2, 1], title="Total Trigram Count")

for i in range(3):
    for ii in range(2):
        ax[i, ii].set_xticklabels(stats.index, rotation=0)
        ax[i, ii].set_xlabel("Processing step")
        ax[i, ii].set_ylabel("Counts")
        ax[i, ii].set_axisbelow(True)
        ax[i, ii].grid(axis="y")

fig.tight_layout()

print('\n\nSaving figure to "ngram_counts.png"\n\n')
fig.savefig("ngram_counts.png")


# In[80]:


# Print top unigrams, bigrams, and trigrams
for index, row in stats.iterrows():
    print(f"Processing step: {index}")
    print("Top unigrams:")
    print([token[0] for token in row["Top_Unigrams"]])
    print("Top bigrams:")
    print([token[0] for token in row["Top_Bigrams"]])
    print("Top trigrams:")
    print([token[0] for token in row["Top_Trigrams"]])
    print("\n")

