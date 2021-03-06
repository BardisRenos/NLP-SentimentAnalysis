# NLP-SentimentAnalysis

<p align="center"> 
<img src="https://github.com/BardisRenos/NLP-SentimentAnalysis/blob/master/images.png" width="300" height="300" style=centerme>
</p>

## Intro 

Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine. More information you can find from this [link](https://en.wikipedia.org/wiki/Sentiment_analysis).


## Regarding this repo

This repo will provide a small example of how a **Recurrent Neural Network (RNN)** using the **Long Short Term Memory (LSTM)** architecture can be implemented using **Keras**. The model will use the data set of IMDB dataset.


## Description of the data

IMDB dataset having 50K movie reviews for natural language processing or Text analytics. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms. For more dataset information, please go through the following [link](http://ai.stanford.edu/~amaas/data/sentiment/)


## Recurrent Neural Network (RNN) Model
<p align="justify"> 
A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs.
 </p>
 
 <p align="center"> 
<img src="https://github.com/BardisRenos/NLP-SentimentAnalysis/blob/master/RNN.png" width="300" height="200" style=centerme>
</p>
 
 

## LSTM Architecture
<p align="justify"> 
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDSs (intrusion detection systems).
 </p>
 
<p align="center"> 
<img src="https://github.com/BardisRenos/NLP-SentimentAnalysis/blob/master/lstm.png" width="300" height="300" style=centerme>
</p>

## Structure of the data set

```python
 import pandas as pd

 df = pd.read_csv('/home/renos/Desktop/IMDB Dataset.csv')
 print(df.count())
 print(len(df.columns))
```

```text

 review       50000
 sentiment    50000
 dtype: int64

 The number of the columns are : 2
```


## Text preprocessing

Our text preprocessing will include the following steps:
* Convert all text to lower case.
* Replace REPLACE_BY_SPACE_RE symbols by space in text.
* Remove symbols that are in BAD_SYMBOLS_RE from text.
* Remove “x” in text.
* Remove stop words.
* Remove digits in text.

```python
 def tokeniz_with_spacy(text):
    # Retrieving the column with all reviews
    token_text = parser(text)
    # Lemmatizing each token from the above setence
    token_text = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in token_text]
    # Removing stop words
    token_text = [word.strip(".") for word in token_text if word not in stop_words and word not in punctuations
                  and word.isalnum()]
    # Removing empty strings
    token_text = [word for word in token_text if len(word) > 1]

    return token_text

```

Creating the data training and testing

```python
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
 print(X_train.shape, y_train.shape)
 print(X_test.shape, y_test.shape)
```


## Creating the deep learning model

```python
 model = Sequential()
 model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
 model.add(SpatialDropout1D(0.2))
 model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
 model.add(Dense(2, activation='softmax'))
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

```
Setting the other parameter of the model

```python
 epochs = 3
 batch_size = 128

 print(model.summary())

 history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=True)
 accuracy = model.evaluate(X_test, y_test)
 print('Test set\n  Loss: {:0.2f}\n  Accuracy: {:0.2f}'.format(accuracy[0], accuracy[1]))

```

```python
 # Plot model accuracy over epochs
 import seaborn as sns
 sns.reset_orig() 
 plt.plot(history.history['acc'])
 plt.plot(history.history['val_acc'])
 plt.title('model accuracy')
 plt.ylabel('accuracy')
 plt.xlabel('epoch')
 plt.legend(['train', 'valid'], loc='upper left')
 plt.show()

 # Plot model loss over epochs
 plt.plot(history.history['loss'])
 plt.plot(history.history['val_loss'])
 plt.title('model loss')
 plt.ylabel('loss')
 plt.xlabel('epoch')
 plt.legend(['train', 'valid'], loc='upper left')
 plt.show()

```

 
<p align="center"> 
<img src="https://github.com/BardisRenos/NLP-SentimentAnalysis/blob/master/download1.png" width="300" height="300" style=centerme>
</p>

<p align="center"> 
<img src="https://github.com/BardisRenos/NLP-SentimentAnalysis/blob/master/download2.png" width="300" height="300" style=centerme>
</p>
