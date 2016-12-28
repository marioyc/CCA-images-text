from gensim.models import word2vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences, size=200)
model.save('text8.model')
model.save_word2vec_format('text.model.bin', binary=True)
