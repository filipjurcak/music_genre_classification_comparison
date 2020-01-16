# Music Genre Classification Methods Performance on Different-Sized Datasets

 Hi everyone, this is my Github repo for project which was a part of Machine Learning class at Comenius University
 in Bratislava
 
 In this project, I wanted to look at what are the best common music genre classification methods and test how well 
 they will perform on larger dataset than they were trained and validated on
 
 ## Datasets
 For two different sized datasets I chose [GTZAN dataset](http://marsyas.info/downloads/datasets.html) and
 [FMA Small dataset](https://github.com/mdeff/fma#history) because they both consists of audio files rather than just
 metadata. You can download them if you want to train models on different features from the tracks in them. In that case
 just put downloaded directories, run `move_files_to_genre_dir.py` and after that you would also want to run the
 `preprocessing.py`, which will do feature extraction for you, so please feel free to change it to your liking
 (can take several hours to generate files). In case you want to go with already provided features, you can either find
 `csv` files in `gtzan` and `fma_small` directories or in extracted `.npy` melspectogram files.
 
 ## Models
 I chose 5 different models for classification, namely:
 * K-Nearest Neighbors (K-NN)
 * SVM
 * Random Forrest ()
 * Neural Network (NN)
 * Convolutional Neural Network (CNN)
 
 All of these models are in `models.py` file, you can also adjust them to your needs.
 
 ## Evaluation
 Each method was evaluated on both datasets, summarization of test accuracy of all models is in the table below:
 
| Model          | GTZAN | FMA Small |
|:--------------:|:-----:| :--------:|
| K-NN           | 60%   | 45.7%     |
| SVM            | 64%   | 46.8%     |
| Random Forrest | 64%   | 45.8%     |
| Neural Network | %    |  %        |
| CNN            | %    |  %        |

As you can see, there is quite a drop-off in accuracy when a model is trained on a bigger dataset than it was initially
optimized for. This indicates that the original dataset wasn't large enough to account for the real error of the models. 
