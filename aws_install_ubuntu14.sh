apt-get update
apt-get upgrade

apt-get install -y python2.7-dev python-pip libblas-dev liblapack-dev gfortran

pip install numpy scipy
pip install scikit-learn

pip install gensim
#wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
wget http://mattmahoney.net/dc/text8.zip
apt-get install -y unzip
unzip text8.zip
python load_word2vec.py
