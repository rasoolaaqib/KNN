#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import distance
import statistics
import random


# In[4]:


#reading and storing train data
train = pd.read_csv("train.csv")  
#reading and storing test data
test = pd.read_csv("test.csv")
##reading and storing stop words
f = open('stop_words.txt', 'r+')  
stop_words = f.read().splitlines()


# In[4]:


def word_split(sent): #A function to split a string into a list of words
    words = re.sub("[\W]"," ", sent).split()
    return words


# In[8]:


def clean_data(x):
  #converting to lowercase
  x = x.apply(lambda x: x.astype(str).str.lower())
  #removing stop words
  for i in stop_words : 
      x = x.replace(to_replace=r'\b%s\b'%i, value="",regex=True)
  #removing punctuations
  table = str.maketrans(dict.fromkeys(string.punctuation))
  for index, value in x['Tweet'].items():
      x['Tweet'][index]=x['Tweet'][index].translate(table)
  #removing numbers
  x = x.replace(to_replace=r'\d', value="",regex=True)
  return x


# In[5]:


def knn(k):
  p=0
  s=distances[0].size
  print("k = ", k)
  perdicted_label = []
  while p<s:
    t=k
    dist = sorted(distances.iloc[p])
    #print(minimum_dist)
    while t>0:
      #print(k,"   ",t)
     ##print("KK=",k)
      index_distance = []
      n=0
      for x, y in enumerate(distances.iloc[p]):
        if n==t:
          #print(n)
          break
        if dist[n]==y:
          index_distance.append(x)
          n+=1
      #print(labels[index_distance])
      try:
        xx = statistics.mode(labels[index_distance])
      except:
        #print(index_distance)
        #print("Multiple modes for k = ", t)
        t-=1
        continue
      else:
        #print(xx)
        perdicted_label.append(xx)
        #print(perdicted_label)
        p+=1
        break
  return perdicted_label


# In[6]:


def measures(perdicted_label,accuracy,precision,f1,recall):
  #Accuracy Calculation
  correct = 0
  for x,y in enumerate(perdicted_label):
    if y == test['Sentiment'][x]:
      correct+= 1
  Accu = (correct/test['Sentiment'].size)
  accuracy.append(Accu)
  print("Accuracy: ", Accu)
  #calculating false and true positives, negatives, and neutrals
  p_pos=0
  p_neg=0
  p_nut=0
  n_pos=0
  n_neg=0
  n_nut=0
  nu_pos=0
  nu_neg=0
  nu_nut=0
  for x,y in enumerate(perdicted_label):
    if   (y == "positive") & (test['Sentiment'][x] == "positive"):  p_pos+=1
    elif (y == "positive") & (test['Sentiment'][x] == "negative"):  p_neg+=1
    elif (y == "positive") & (test['Sentiment'][x] == "neutral"):   p_nut+=1
    elif (y == "negative") & (test['Sentiment'][x] == "negative"):  n_neg+=1
    elif (y == "negative") & (test['Sentiment'][x] == "positive"):  n_pos+=1
    elif (y == "negative") & (test['Sentiment'][x] == "neutral"):   n_nut+=1
    elif (y == "neutral") & (test['Sentiment'][x] == "positive"):   nu_pos+=1
    elif (y == "neutral") & (test['Sentiment'][x] == "negative"):   nu_neg+=1
    elif (y == "neutral") & (test['Sentiment'][x] == "neutral"):    nu_nut+=1
  #calculating macroaverage recall
  pos_recall= p_pos/(p_pos+p_neg+p_nut)
  neg_recall= n_neg/(n_pos+n_neg+n_nut)
  nut_recall= nu_nut/(nu_pos+nu_neg+nu_nut)
  macro_avg_recall = (pos_recall+neg_recall+nut_recall)/3
  recall.append(macro_avg_recall)
  #calculating macroaverage precision
  pos_precision= p_pos/(p_neg+p_pos+p_nut)
  neg_precision= n_neg/(n_neg+n_pos+n_nut)
  nut_precision= nu_nut/(nu_neg+nu_pos+nu_nut)
  macro_avg_precision = (pos_precision+neg_precision+nut_precision)/3
  precision.append(macro_avg_precision)
  #calculating macroaverage F1-score
  F1_score = (2*macro_avg_precision*macro_avg_recall)/(macro_avg_recall+macro_avg_precision)
  f1.append(F1_score)
  #outputing macroaverage recall precision and F1-Score
  print("Macroaverage Recall: ", macro_avg_recall)
  print("Macroaverage Percision: ", macro_avg_precision)
  print("F1 Score: ", F1_score)
  #Building a confusion matrix
  print("Confusion Matrix: ")
  conf = {'Outputs/Gold Labels' : ['positive','neutral','negative'], 'positive' : [p_pos,p_nut,p_neg], 'neutral' : [nu_pos,nu_nut,nu_neg], 'negative' : [n_pos,n_nut,n_neg]}
  confusion=pd.DataFrame(conf, columns= ['Outputs/Gold Labels', 'positive', 'neutral', 'negative'])
  print(confusion)


# In[9]:


train = clean_data(train)
test = clean_data(test)


# In[11]:


#converting to lowercase
train = train.apply(lambda x: x.astype(str).str.lower())
#removing stop words
for i in stop_words : 
    train = train.replace(to_replace=r'\b%s\b'%i, value="",regex=True)
#removing punctuations
table = str.maketrans(dict.fromkeys(string.punctuation))
for index, value in train['Tweet'].items():
    train['Tweet'][index]=train['Tweet'][index].translate(table)
#removing numbers
train = train.replace(to_replace=r'\d', value="",regex=True)


# In[65]:



#converting to lowercase
test = test.apply(lambda x: x.astype(str).str.lower())
#removing stop words
for i in stop_words : 
    test = test.replace(to_replace=r'\b%s\b'%i, value="",regex=True)
#removing punctuations
table = str.maketrans(dict.fromkeys(string.punctuation))
for index, value in test['Tweet'].items():
    test['Tweet'][index]=test['Tweet'][index].translate(table)
#removing numbers
test = test.replace(to_replace=r'\d', value="",regex=True)


# In[10]:


vocabulary= [] #building vocabulary series for bag of words from train data
for x in train['Tweet'].tolist():
    a = word_split(x)
    vocabulary.extend(a);
vocabulary = list(set(vocabulary))
vocab=pd.Series(vocabulary)
dup_vocab = vocab


# In[11]:


#building bag of words with vocabulary as columns and tweets as rows from train data
bow = pd.DataFrame (columns=dup_vocab)
ss = len(dup_vocab)
for x in train['Tweet']:
    c = word_split(x)
    bow_vector = np.zeros(ss)
    for d in c:
        for i, y in enumerate(dup_vocab):
            if y==d:
                bow_vector[i] = bow_vector[i] + 1
    #a = pd.DataFrame([bow_vector],columns = dup_vocab)
    bow = bow.append(pd.Series(bow_vector, index=dup_vocab),ignore_index=True)
    #print(bow.shape)
dup_bow = bow


# In[12]:


distances= []
#building bag of words with vocabulary as columns and tweets as rows from test data
test_bow = pd.DataFrame (columns=vocab)
for x in test['Tweet'].tolist():
    c = word_split(x)
    test_bow_vector = np.zeros(ss)
    for d in c:
      for i, y in enumerate(vocab):
        if y==d:
          test_bow_vector[i] = test_bow_vector[i] + 1
    #a = pd.DataFrame([test_bow_vector],columns = dup_vocab)
    test_bow = test_bow.append(pd.Series(test_bow_vector, index=vocab),ignore_index=True)
    #print(test_bow.shape)
    #print(test_bow)
test_dup_bow = test_bow


# In[ ]:


#finding euclidean distances
distances = scipy.spatial.distance.cdist(test_dup_bow.values, dup_bow.values, metric='euclidean')


# In[ ]:


#distances as a dataframe
distances = pd.DataFrame(distances)


# In[ ]:


labels=train['Sentiment'] #train data labels (Gold labels)
accuracy = []
percision = []
f1 = []
recall= []
p_label = knn(10) #predicting labels for test data for k=10
measures(p_label,accuracy,percision,f1,recall) #measuring accuracy, confusion matrix, precision, f1-score, and recall 
p_label = knn(7) #predicting labels for test data for k=7
measures(p_label,accuracy,percision,f1,recall) #measuring accuracy, confusion matrix, precision, f1-score, and recall 
p_label = knn(5) #predicting labels for test data for k=5
measures(p_label,accuracy,percision,f1,recall) #measuring accuracy, confusion matrix, precision, f1-score, and recall 
p_label = knn(3) #predicting labels for test data for k=3
measures(p_label,accuracy,percision,f1,recall) #measuring accuracy, confusion matrix, precision, f1-score, and recall 
p_label = knn(1) #predicting labels for test data for k=
measures(p_label,accuracy,percision,f1,recall) #measuring accuracy, confusion matrix, precision, f1-score, and recall 


# In[ ]:


k= [10,7,5,3,1]
#Plotting graphs for accuracy, precision, recall, and f1-score against all k values
fig, axs = plt.subplots(2,2)
axs[0,0].plot(k,accuracy)
axs[0,0].set_title('Accuracy')
axs[1,0].plot(k,percision)
axs[1,0].set_title('Precision')
axs[1,1].plot(k,recall)
axs[1,1].set_title('Recall')
axs[0,1].plot(k,f1)
axs[0,1].set_title('F1 Score')
for ax in axs.flat:
    ax.set(xlabel='k-values')

