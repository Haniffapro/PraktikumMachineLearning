# PraktikumMachineLearning

#Create your own API from kaggle and set to your directory
!mkdir -p ~/.kaggle

#kaggle.json from kaggle API
!cp kaggle.json ~/.kaggle/

#download datasets
!kaggle datasets download chetankv/dogs-cats-images

#unzip datasets
!unzip "dogs-cats-images"
