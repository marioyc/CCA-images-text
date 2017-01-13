apt-get update
apt-get upgrade

apt-get install -y python2.7-dev python-pip libblas-dev liblapack-dev gfortran python-matplotlib

pip install numpy scipy
pip install progressbar2
#pip install scikit-learn

pip install gensim nltk
python -m nltk.downloader punkt
#wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
wget http://mattmahoney.net/dc/text8.zip
apt-get install -y unzip
unzip text8.zip
rm text8.zip
python load_word2vec.py

apt-get install -y libhdf5-dev
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64-deb
dpkg -i cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64-deb
rm cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64-deb
apt-get install -y cuda
# need to download the Runtime Library and Developer Libray from https://developer.nvidia.com/cuDNN
sudo dpkg -i libcudnn5_5.1.5-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn5-dev_5.1.5-1+cuda8.0_amd64.deb
echo "\nexport CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "\nexport PATH=$PATH:/usr/local/cuda/bin" >> ~/.bashrc
echo "\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl

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
