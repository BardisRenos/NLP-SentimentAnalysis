import string
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

# import the spaCy library
import spacy
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split

# Setting the language
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()
# Create our list of punctuation marks
punctuations = string.punctuation

# Creating a pipeline to .csv file
df = pd.read_csv('/home/renos/Desktop/IMDB Dataset.csv')


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


df['review'] = df['review'].apply(tokeniz_with_spacy)
max_review_length = 500
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
print(df['review'])
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['review'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['review'].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
y = pd.get_dummies(df['sentiment']).values
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

epochs = 3
batch_size = 64

print(model.summary())

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=True)
accuracy = model.evaluate(X_test, y_test)
print('Test set\n  Loss: {:0.2f}\n  Accuracy: {:0.2f}'.format(accuracy[0], accuracy[1]))

# Plot model accuracy over epochs
import seaborn as sns
sns.reset_orig()  # Reset seaborn settings to get rid of black background
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
