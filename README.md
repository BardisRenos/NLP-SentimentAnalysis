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
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition,[2] speech recognition[3][4] and anomaly detection in network traffic or IDSs (intrusion detection systems).
 </p>
 
 <p align="center"> 
<img src="https://github.com/BardisRenos/NLP-SentimentAnalysis/blob/master/lstm.png" width="300" height="300" style=centerme>
</p>

## Structure of the data set

```python
 import pandas as pd

 df = pd.read_csv('/home/renos/Desktop/IMDB Dataset.csv')
 # print(df['review'])
 # print(df[['Title', 'ID']])
 # print(df['product'].value_counts().count())
 print(df.count())
 print(len(df.columns))
```

```text

 review       50000
 sentiment    50000
 dtype: int64

```

