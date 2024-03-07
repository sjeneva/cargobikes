import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk

# Ensuring the necessary NLTK data is available; uncomment if needed
# nltk.download('stopwords')

# List of document titles on urban logistics and cargo bikes
documents = [
    # Add all your document titles here
]

# Load stop words from NLTK
stop_words = stopwords.words('english')

# Preprocess documents: tokenize and remove stop words
preprocessed_docs = [[word for word in simple_preprocess(doc) if word not in stop_words] for doc in documents]

# Create a Dictionary from the preprocessed documents
id2word = corpora.Dictionary(preprocessed_docs)

# Create a Corpus: a list of Bag-of-Words (BoW) for each document
corpus = [id2word.doc2bow(text) for text in preprocessed_docs]

# Build the LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=id2word,
                     num_topics=2,  # Assuming you want to identify 2 topics
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# Display the topics identified by the LDA model
topics = lda_model.print_topics()
for topic in topics:
    print(topic)
