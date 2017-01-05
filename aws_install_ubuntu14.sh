apt-get update
apt-get upgrade

apt-get install -y python2.7-dev python-pip libblas-dev liblapack-dev gfortran python-matplotlib

pip install numpy scipy
pip install scikit-learn

pip install gensim nltk
python -m nltk.downloader all
#wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
wget http://mattmahoney.net/dc/text8.zip
apt-get install -y unzip
unzip text8.zip
rm text8.zip
python load_word2vec.py

apt-get install -y libhdf5-dev
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
pip install keras
pip install h5py

wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
unzip train2014.zip
unzip val2014.zip
unzip captions_train-val2014.zip
rm train2014.zip
rm val2014.zip
rm captions_train-val2014.zip

pip install Cython
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
python setup.py build_ext install
