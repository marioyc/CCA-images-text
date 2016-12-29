from gensim.models import word2vec

model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)

print model.most_similar(positive=['woman', 'king'], negative=['man'], topn=2)

print model.most_similar(['man'])

print model.most_similar(['girl', 'father'], ['boy'], topn=3)

v = model['man']

print v.size, len(v), v[0]

print model.similarity('woman', 'man')

print len(model.vocab)

cont = 0
for k in model.vocab.keys():
    print k
    cont += 1
    if cont == 5:
        break
