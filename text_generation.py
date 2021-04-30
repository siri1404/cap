#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# In[2]:


#load data
file = open("frankenstein-2.txt").read()


# In[3]:


#tokenization
#standardization
def tokenize_words(input):
    input = input.loower()
    tokenizer  = RegexpTokenizer(r'\wt')
    tokens = tokenizer.tokenize(input)
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return "".join(filtered)

processes_inputs = tokenize_words(file)


# In[4]:


#chars to numbers
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i, c in enumerate(chars))


# In[5]:


#check if words to chars or chars to num (?!) has worked?
#just so we get an idea of whether our process of converting words to characters has worked
#we print the length of the variables
input_len = len(processed_inputs)
vocab_len = len(chars)
print("Total number of characters:", input_len)
print("Total vocab:", vocab_len)


# In[6]:


#seq length
#we're defining how long we want an indiviual sequence here
#an indiviual sequence is a complete mapping of input characters as integers
seq_length = 100
x_data = []
y_data = []


# In[7]:


# loop through the sequence
#here we're going through the eentire list of i/ps and converting the chars to number with a for loop
#this will create a bunch of sequences where each sequence starts with the next character in the i/p data
#beginning with the first character
for i in range(0, input_len - seq_length, l):
    in_seq = processed_inputs[i:i + seq_length]
    out_seq = processed_inputs[i + seq_length]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append([char_to_num[out_seq]])


# In[8]:


n_patterns = len(x_data)
print("Total Patterns:", n_patterns)


# In[9]:


#convert input sequence to np array and so on
x = numpy.reshape(x_data, (n_patterns, seq_length, l))
x = x/float(vocab_len)


# In[10]:


# one-hot encoding
y = np_utils.to_categorical(y_data)


# In[11]:


#creating the model
model = sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


# In[12]:


#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[13]:


# saving weights
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, mnitor='loss', verbose = 1, save_best_only=True, mode='min')
desired_cllbacks=[checkpoint]


# In[14]:


#fit model and let it train
model.fit(x,y, epochs=4, batch_size=256, callbacks=desired_callbacks)


# In[15]:


#recompile the model with the saved weights
filename="model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[16]:


#output of the model back into characters
num_to_char = dict((i,c) for i,c in enumerate(chars))


# In[17]:


#random seed to help generate
start = numpy.random.randint(0, len(x_data)-1)
pattern = x_data(start)
print("Random Seed: ")
print("\"",''.join([num_to_char[value] for value in pattern]), "\"")


# In[19]:


#generate the text 
for i in range(1000):
    x = numpy.reshape(pattern, (l,len(pattern), l))
    x = x/float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[l:len(pattern)]


# In[ ]:




