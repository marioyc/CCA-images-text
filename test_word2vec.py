from gensim.models import word2vec

model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)

print model.most_similar(positive=['woman', 'king'], negative=['man'], topn=2)

print model.most_similar(['man'])

print model.most_similar(['girl', 'father'], ['boy'], topn=3)
