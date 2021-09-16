import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import nltk.corpus
from nltk.corpus import stopwords




#import data set 
df = pd.read_csv('covid fake.csv')

#check for missing data and column names
msno.matrix(df)
df.head()

#drop irrelevent column 
df = df.drop(['Unnamed: 0'], axis=1)

msno.matrix(df)
df.head()


#Check which rows are missing 

df.count()

df.isnull().any(axis = 1).sum()


NaN_instances = df[df.isnull().any(axis=1)]
NaN_instances

#show entire instance 
pd.set_option("display.max_colwidth", None)

#drop missing rows 
df = df.drop(labels=[84,1375])

#check missing rows have been removed 
df.isnull().any(axis = 1).sum()

df.count()

#reset index
df = df.reset_index(drop=True)


#count number of words per feature and add two new columns 
df['total words of text'] = [len(x.split()) for x in df['text'].tolist()] # without counting punctuation 
df['total words of title'] = [len(x.split()) for x in df['title'].tolist()] # without counting punctuation


#create temp df to carry out visualization 
temp = df.groupby('label').count()
temp

#pie chart class frequency count
plt.figure()
labels = ['Fake','Real']
plt.pie(temp['title'],labels = labels, autopct = '%.2f%%')
plt.title('Class frequency count')
plt.show()


#hisotgram of the distribution text length  
word_count = [len(i.split()) for i in df['text']]

plt.figure()
pd.Series(word_count).hist(bins = 1000,color='blue')
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')
plt.title('distribution of text length')

#remove outliers 
temp = df.loc[df['total words of text'] >= 3000]
temp

df = df.drop(labels=[4,157,267,526,574,667,937,1244,1464,1513,1535,1794, 1806,
                    2070,2078, 2080, 2185, 3108, 2249, 2285, 2549, 2560, 2668, 2905, 2990])
df = df.reset_index(drop=True)

#histogram of the distribution of title length 

word_count = [len(i.split()) for i in df['title']]

plt.figure()
pd.Series(word_count).hist(bins = 33,color='blue')
plt.xlabel('Number of Words')
plt.ylabel('Number of titles')
plt.title('distribution of title length')


#remove outliers 
temp = df.loc[df['total words of title'] >= 25]
temp

df = df.drop(labels=[173, 204, 374, 468, 491, 588, 591, 641, 719, 726, 
                    1307, 1373, 1403, 1486, 1504, 1516, 1626, 1627, 1738, 1874, 1931, 3022,
                     1961, 1997, 2358, 2376, 2462, 2642, 2694, 2698, 2726])
df = df.reset_index(drop=True)

#check changes were made

#hisotgram of the distribution text length  
word_count = [len(i.split()) for i in df['text']]

plt.figure()
pd.Series(word_count).hist(bins = 1000,color='blue')
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')
plt.title('distribution of text length')

df = df.drop(labels=[3045])
df = df.reset_index(drop=True)

#histogram of the distribution of title length 

word_count = [len(i.split()) for i in df['title']]

plt.figure()
pd.Series(word_count).hist(bins = 33,color='blue')
plt.xlabel('Number of Words')
plt.ylabel('Number of titles')
plt.title('distribution of title length')

#create temp df to carry out visualization 
temp = df.groupby('label').count()
temp

#pie chart class frequency count - remains virtually the same 
plt.figure()
labels = ['Fake','Real']
plt.pie(temp['title'],labels = labels, autopct = '%.2f%%')
plt.title('Class frequency count')
plt.show()

#average word counts per class
temp = df.groupby(['label']).mean()
temp.reset_index(inplace=True)

temp['label'].replace({0:'Fake news', 1:'Real news'}, inplace=True)

#bar plot - title
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

x = temp['label']
y = temp['total words of title']
ax.bar(x,y)
ax.set_title('Average words per title for fake and genuine news')
ax.set_ylabel('Mean average words in title')
ax.set_xlabel('News')

plt.show()

#bar plot - text
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

x = temp['label']
y = temp['total words of text']
ax.bar(x,y)
ax.set_title('Average words per text for fake and genuine news')
ax.set_ylabel('Mean average words in text')
ax.set_xlabel('News')

plt.show()

#remove unnecessary columns 

df = df.drop(['subcategory','total words of text', 'total words of title'],axis = 1)
df.head()

#creat x and y 
X = df.drop(['label'], axis = 1)
y = df['label']

#sampling methods 

#change over to under to change sampling method 
R_O_S= RandomOverSampler()
# resampling 
X_OS, y_OS = R_O_S.fit_resample(X, y)
# new class distribution 
print(Counter(y_OS))

y_OS = y_OS.to_frame(name='label')


#merge title with text 
X_OS['text']=X_OS['title']+X_OS['text']

X_OS['label'] = y_OS['label']

#merge title with text - drop title column no longer needed
df = X_OS
df= df.drop(['title'], axis = 1)


#data preprocessing 


#lower case 
df['text'] = df['text'].str.lower()

#remove numbers 

#the \d stands for 'anyt digit' and the + stands for one or more 
#this function  "Replace all occurring digits in the strings with nothing".

df['text'] = df['text'].str.replace('\d+', '')

#remove punctuation 
df['text'] = df['text'].str.replace(r'[^\w\s]+', '')


#stem words
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter_stemmer = PorterStemmer()
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['text'] = df['text'].apply(stem_sentences)

#remove stop words
nltk.download('stopwords')
stop_words = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#RNN
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, RNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

#count unique words for max words 
count = Counter()
def word_count(text):
    for o in text.values:
        for WD in o.split():
            count[WD] += 1
    return count 

counter = word_count(df['text'])

#number of words 
max_words = len(counter)
# maximum number of words in a specific sequence 
max_len = 3050

#split data 
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size = 0.20, shuffle = True)

#tokenize train data set
token = Tokenizer(num_words=max_words, lower=True, split=' ') 
token.fit_on_texts(x_train.values)            

#pad sequences to specific length for train and test data
sequences = token.texts_to_sequences(x_train.values)                       
train_sequences_padded = pad_sequences(sequences, maxlen=max_len)

test_sequences = token.texts_to_sequences(x_test)
test_sequences_padded = pad_sequences(test_sequences, maxlen=max_len)


#Baseline model
embed_dim = 100
lstm_out = 128
batch_size = 64

model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length = max_len))
model.add(LSTM(lstm_out))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer= 'adam' ,metrics = ['accuracy'])


#train and test 
train_model = model.fit(train_sequences_padded, y_train, batch_size=batch_size, epochs = 100,
                        shuffle = True, verbose =True, validation_split=0.2)

test_model = model.evaluate(test_sequences_padded, y_test)



y_pred = model.predict_classes(test_sequences_padded)

#confusion matrix
confusion_matrix(y_test, y_pred)


y_predict = model.predict_classes(test_sequences_padded)

# baseline results
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print(f'Accuracy Score : {(round(accuracy_score(y_test,y_predict)*100,2))}%')
print(f'Precision Score : {str(round(precision_score(y_test,y_predict)*100, 2))}%')
print(f'Recall Score : {str(round(recall_score(y_test,y_predict)*100,2))}%')
print(f'F1 Score : {(round(f1_score(y_test,y_predict)*100,2))}%')


#parameter tuning
Dense1 = [25,50,75,100]
Dense2 = [100,200,500,1000]
Dropout1 = [0.5,0.7,0.9]
optimizer = ['adam','Adamax' , 'Nadam', 'RMSprop'] 

loss = [] # loss
ll = [] #accuracy
#parameter 
Dense_1 = [] # Dense1 
Dense_2 = [] # Dense2
Dropout2 = [] # Dropout1
optimizer1 = [] # optimizer
history1 = [] # history 


for x in Dense1:
    for i in Dense2:
        for u in Dropout1:
            for e in optimizer:
                model = Sequential()
                model.add(Embedding(max_words, embed_dim, input_length = max_len))
                model.add(LSTM(lstm_out))
                model.add(Dense(x, activation = 'relu'))
                model.add(Dense(i, activation='relu'))
                model.add(Dropout(u))
                model.add(Dense(x, activation='relu'))
                model.add(Dense(1, activation = 'sigmoid'))
                model.compile(loss = 'binary_crossentropy', optimizer= e ,metrics = ['accuracy'])
                print(f'Dense layer 1 and 3 = {x}')
                print(f'Dense layer 2 = {i}')
                print(f'Dropout = {u}')
                print(f'optimizer = {e}')
                history = model.fit(train_sequences_padded, y_train, batch_size=batch_size, epochs = 100,
                                    shuffle = True, verbose = True, validation_split=0.2)
                test = model.evaluate(test_sequences_padded, y_test)
                print(f'Dense 1+3 = {x} Dense 2 = {i} Dropout = {u} optimizer = {e}') #check tuning is working 
                loss.append(test[0])
                ll.append(test[1])
                print('ll',ll)
                Dense_1.append(x)
                Dense_2.append(i)
                Dropout2.append(u)
                optimizer1.append(e)
                history1.append(history)
                

#optimised model 
max_score = max(ll)
max_score = float(max_score)
index_num = ll.index(max_score) #optimal parameter index 


#optimal parameters 
print(f'Hyper-parameters are:')
print(f'Dense layer 1 and 3 are: {Dense_1[index_num]}')
print(f'Dense layer 2 is: {Dense_2[index_num]}')
print(f'Dropout is : {Dropout2[index_num]}')
print(f'Dense optimiser is: {optimizer1[index_num]}')

#optimal model
model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length = max_len))
model.add(LSTM(lstm_out))
model.add(Dense(Dense_1[index_num], activation = 'relu'))
model.add(Dense(Dense_2[index_num], activation='relu'))
model.add(Dropout(Dropout2[index_num]))
model.add(Dense(Dense_1[index_num], activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer= optimizer1[index_num] ,metrics = ['accuracy'])

#train
history = model.fit(train_sequences_padded, y_train, batch_size=batch_size, epochs = 100,
                    shuffle = True, verbose =True, validation_split=0.2)
#test
test = model.evaluate(test_sequences_padded, y_test)

#confusion matrix 

y_pred = model.predict_classes(test_sequences_padded)

confusion_matrix(y_test, y_pred)

#optimised results 
y_predict = model.predict_classes(test_sequences_padded)

print(f'Accuracy Score : {(round(accuracy_score(y_test,y_predict)*100,2))}%')
print(f'Precision Score : {str(round(precision_score(y_test,y_predict)*100, 2))}%')
print(f'Recall Score : {str(round(recall_score(y_test,y_predict)*100,2))}%')
print(f'F1 Score : {(round(f1_score(y_test,y_predict)*100,2))}%')

#tuning results - line graph

ypoints = np.array(ll)
plt.figure(figsize = (10,5))
plt.plot(ypoints, linestyle = 'solid', color = 'r', linewidth = 1)
plt.xlabel("RNN model")
plt.ylabel("accuracy score")
plt.title('Mean average scores')
plt.show()

ypoints = np.array(ll)
plt.figure(figsize = (10,5))
plt.plot(ypoints, linestyle = 'solid', color = 'r', linewidth = 1)
plt.xlabel("RNN model")
plt.ylabel("accuracy score")
plt.title('Mean average scores')
plt.show()
