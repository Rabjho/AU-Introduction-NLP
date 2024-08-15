#!/usr/bin/env python
# coding: utf-8

# # Method
# 1. Load in data
#    1. Pre-process data
# 2. Split into train and test data
# 3. Extract features and train classifier
# 4. Optional: Inspect classifier
# 5. Run classifier on held-out test set
# 6. Calculate precision and recall
# 
# # Idea(s)
# 1. Compare Multinomial Naive Bayes with binary version.
# 2. Compare dealing with negation by prepending "NOT_" until punctuation
# 3. Compare using full vocabulary or only using words occuring more times than some threshold(s)
# 4. Increase computational efficiency stemming, removing stopwords etc.
# 5. Compare weighting tokens using tf-idf or PPMI against not.
# 6. Train some other models
#    1. Naive Bayes
#    2. Binary Naive Bayes
#    3. LSTM

# In[1]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk.downloader
nltk.download("punkt")

# Import imdb data
imdb = pd.read_csv('./data/imdb/imdb_dataset.csv')

# Tokenization
imdb["tokens"] = imdb["review"].apply(word_tokenize)


# In[2]:


# Set some amount of the dataset to be used for training and testing
test_size = 0.1 # Amount of data to be used for testing

# Split the data into training and testing data
imdb["type"] = "train"
imdb.loc[imdb.sample(frac=test_size).index, "type"] = "test"


# In[3]:


# Create bag-of-words representation of each document (using a simple Counter)
from collections import Counter
imdb["bow"] = imdb["tokens"].apply(Counter)

# Create a bow representation of the entire dataset
bow = Counter()
for counter in imdb["bow"]:
    bow.update(counter)

# Find vocabulary with possibility to remove words that are too rare or too common
occurance_threshold = 0 # Possibility of a threshold to make runtime faster. Set to 0 to include all words
vocabulary = set()
for word, count in bow.items():
    if count > occurance_threshold:
        vocabulary.add(word)
        
# Filter bow of each document to only include words from the vocabulary
imdb["filtered_bow"] = imdb["bow"].apply(lambda counter: {word: count for word, count in counter.items() if word in vocabulary})

# Report how many words are filtered out
print(f"Filtered out {len(bow) - len(vocabulary)} words")


# In[4]:


from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(imdb[imdb["type"] == "train"][["filtered_bow", "sentiment"]].values)

imdb.loc[imdb["type"] == "test", "predictions"] = classifier.classify_many(imdb[imdb["type"] == "test"]["filtered_bow"].values)


# In[5]:


# Calculate accuracy
correct = (imdb.loc[imdb["type"] == "test", "predictions"] == imdb[imdb["type"] == "test"]["sentiment"]).sum()/len(imdb.loc[imdb["type"] == "test", "predictions"])  # Evil sum of booleans hack
correct


# In[6]:


# Calculate precision, recall and F1
def precision_recall_f1(pred_labels, true_labels):
    tp = (pred_labels & true_labels).sum()
    fp = (pred_labels & ~true_labels).sum()
    fn = (~pred_labels & true_labels).sum()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

print(precision_recall_f1(imdb.loc[imdb["type"] == "test", "predictions"] == "positive", imdb.loc[imdb["type"] == "test", "sentiment"] == "positive"))


# In[7]:


# Create a function that takes a review and returns the sentiment
def predict_sentiment(review):
    tokens = word_tokenize(review)
    bow = Counter(tokens)
    filtered_bow = {word: count for word, count in bow.items() if word in vocabulary}
    return classifier.classify(filtered_bow)


# In[12]:


# Find max number of tokens in a review
imdb["tokens"].apply(len).max()


# In[29]:


# Train LSTM model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

train_data = imdb[imdb['type'] == "train"]
test_data = imdb[imdb['type'] == "test"]
# Convert labels to integers, positive = 1, negative = 0
train_data['sentiment'] = (train_data['sentiment'] == 'positive').astype(int)
test_data['sentiment'] = (test_data['sentiment'] == 'positive').astype(int)

tokenizer = Tokenizer(num_words=len(vocabulary)) # Need to tokenize into integers
tokenizer.fit_on_texts(imdb['tokens'].apply(lambda x: ' '.join(x))) # Wait, why am I using tokens and then stitching them together?
X_train_lstm = tokenizer.texts_to_sequences(train_data['tokens'].apply(lambda x: ' '.join(x)))
X_test_lstm = tokenizer.texts_to_sequences(test_data['tokens'].apply(lambda x: ' '.join(x)))

max_len = int(np.percentile([len(seq) for seq in imdb['tokens']], 95))
X_train_lstm = pad_sequences(X_train_lstm, maxlen=max_len)
X_test_lstm = pad_sequences(X_test_lstm, maxlen=max_len)
y_train_lstm = train_data['sentiment']
y_test_lstm = test_data['sentiment']


# In[80]:


import os.path
# Create checkpoints
checkpoint_path = "./sltm.keras"

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1)

def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=len(vocabulary), output_dim=64, input_length=max_len))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)
else:
    # Careful, takes 30 min to run
    model = create_model()
    model.fit(X_train_lstm, y_train_lstm, batch_size=64, epochs=5, validation_data=(X_test_lstm, y_test_lstm), callbacks=[cp_callback])


# In[78]:


y_pred_lstm = model.predict(X_test_lstm)
imdb["predictions_lstm"] = None
imdb.loc[imdb['type'] == 'test', 'predictions_lstm'] = y_pred_lstm > 0.5

print(precision_recall_f1(imdb.loc[imdb['type'] == 'test', 'predictions_lstm'], imdb.loc[imdb['type'] == 'test', 'sentiment'] == 'positive'))


# - (precision, recall, f1) might vary, because the test/training sets are split differently at runtime. These are the ones used in the paper
# - (0.8877772944758591, 0.8253133845531743, 0.8554065381391451) 
# - (0.9110091743119266, 0.8030731904569349, 0.8536428110896196) # The neural network is saved, but the training/test set was not. Subsequent runs will result in false performance

# In[79]:


# Find accuracy of LSTM model
correct_lstm = (imdb.loc[imdb["type"] == "test", "predictions_lstm"] == (imdb[imdb["type"] == "test"]["sentiment"]=="positive")).sum()/len(imdb.loc[imdb["type"] == "test", "predictions"])  # Evil sum of booleans hack
print("Accuracy of LSTM model: ", correct_lstm)


# In[1]:


# Check the distribution of the test data sentiments to see if the model is unbalanced
print(imdb.loc[imdb['type'] == 'test', 'sentiment'].value_counts())

