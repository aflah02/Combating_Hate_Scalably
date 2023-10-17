# Vote! Don’t Retrain: Combating Hate in a Scalable Way

This repository contains our course project for [NLP 2022 Course](http://techtree.iiitd.edu.in/viewDescription/filename?=CSE556). 

*Accepted at ICLR 2023 for the Tiny Papers Track*

### Offensive Speech:
Here, offensive is used as an umbrellla term to determine any form of profane, abusive, hateful, abusive language being used. Offensive speech is any type of speech, conduct, writing, or expression that is considered aggressive, insulting, or degrading to an individual or group of people.

### Associated Task for our Project:
Given a tweet, the task is to identify whether it offensive or not. We deal with labels at the first level, i.e. binary classification of whether a tweet is Hateful/Offensive or not.

# Datasets and Original Tasks
The dataset used in this work is the [Offensive Language Identification Dataset (OLID)](https://aclanthology.org/N19-1144/). The dataset was used in the [SemEval 2019 Competition](https://arxiv.org/abs/1903.08983) where several systems competed to attain the best Macro F1 score. The best performing system finetuned a BERT model with max length 64 and ran the model for 2 epochs and outperformed all other models.

# Methodology
### Data Preprocessing:
We performed the following preprocessing steps on the given tweets ; converting tweets to lowercase, replacing @usernames, urls and emojis with the \<user\>, \<url\> and \<emoji\> token respectively and removing punctuations, stopwards and extra whitespaces.  Then we splitted our given dataset into a training and validation set (80:20).

### Data Visualization:
We tried various visualization techniques to understand the distribution of the task better. We noticed that the data was somewhat skewed, with a majority of the labels in favor of one class (62% NOT Offensive and 38% Offensive) which we can also see in the figure. As with all hate speech datasets, we see a skewed distribution of class labels. However, the skewness in this dataset is not as bad as some other datasets where only around 5-10% of the total dataset is hateful. This prior bias also might contribute to poor results in some cases. We also tried topic modeling (Figures 2, 3) to see if we can see any clear topics across the 2 classes but surprisingly the most popular topics in both datasets are the same. This shows how non-trivial this task is

 <p align="center">
<a>
    <img width="452" alt="image" src="https://user-images.githubusercontent.com/72096386/209498129-ef28464c-c37f-4634-a41f-2ae88969cafa.png">
</a>
 </p>

### Modelling:
We first experimented with classical Machine Learning Models namely SVM, MLP, and Perceptron. For each of these models we convert the input text into sentence vectors by either first obtaining word embeddings and averaging them over the sentence in the case of non contextual embeddings or by directly using a Siamese Network exposed via [SBERT](https://arxiv.org/abs/1908.10084) to obtain the sentence embeddings. We also try RNN, LSTM and GRU based models for our work.

Finally we use a combination of task specific BERT variants for extracting different subspace representations which capture different markers. We use [BERT](https://doi.org/10.18653/v1/N19-1423) to capture high level semantics without any task specific filtering, [HateBERT](http://arxiv.org/abs/2010.12472) to capture signals which are indicative of hatefulness or its absence and [BERTweet](https://doi.org/10.18653/v1/2020.emnlp-demos.2) to capture social media text specific signals. We combine them by concatenating or averaging their last layer representations and only allow the last 3 layers to train to preserve their original skills and avoid catastrophic forgetting. We also experiment with combining them with the [HaT5 RoBERTa model](http://arxiv.org/abs/2202.05690) and MTL model optionally in a max voting fashion.

We also tried combining external knowledge with the help of commonsense or stereotype knowledge graphs, by concatenating the linearised texts of the tuples to the input sentences, but were unfortunately not able to get any good results.

We rank all our models based on the Macro F1 scores obtained as used in the original SemEval task. We believe these scores are not accurate as we report the scores provided by kaggle which only uses 25% of the data. 

For all our simple ML baselines we use Grid Search with 2 Fold Cross Validation to obtain optimal hyperparameters. We use the implementation from SKLearn. For obtaining our non-contextual word embeddings we use the loading wrappers provided by Gensim for [Word2Vec](http://arxiv.org/abs/1301.3781), [GloVe](https://doi.org/10.3115/v1/D14-1162) and [FastText](http://arxiv.org/abs/1607.01759).

# Results
Overall we observe some clear trends. Simple ML Models accompanied by good sentence embeddings can also outperform simple DL baselines. For instance SVM and Perceptron with FastText are better than any other traditional DL Model. MTL, HateBERT and DistilBERT, BERT. HateBERT and TweetBERT all perform well alone however combining and training BERT, HateBERT and BERTweet together does not perform as well. This is infact good for us as this is much more costly than performing max voting which we show yields much higher results. Our Best Max Voting Configuration used HateBERT + MTL Model + HaT5 together. Max Voting simply means take the class which is predicted by a majority of the models.

This yields us 2 interesting results:
* There are some signals which are captured by these separately finetuned language models which makes them stand out. Combining these results brings significant gain to our performance.
* Simple max voting performs much better than fancy heuristics, is scalable and more explainable as we can see which models think the tweet should be classified as hateful and which ones think it should not.

 <p align="center">
<img width="595" alt="image" src="https://user-images.githubusercontent.com/72096386/209498091-de5c0c29-85ee-4de2-b3b0-40c1d140b594.png">
 </p>
 
---
### Baselines and Benchmark Models
Apart from training our own models we also use some benchmark models which acheieved SOTA in SemEval 2019 and 2020

- [Kungfupanda at SemEval-2020 Task 12: BERT-Based Multi-TaskLearning for Offensive Language Detection](https://aclanthology.org/2020.semeval-1.272/)
- [NULI at SemEval-2019 Task 6: Transfer Learning for Offensive Language Detection using Bidirectional Transformers](https://aclanthology.org/S19-2011/)
- [UHH-LT at SemEval-2020 Task 12: Fine-Tuning of Pre-Trained Transformer Networks for Offensive Language Detection](https://aclanthology.org/2020.semeval-1.213.pdf)

For these baselines we use the codes provided by their authors.

Overall our [Final Kaggle contest ranking was 2nd](https://www.kaggle.com/competitions/cse556nlp22projecth1/leaderboard) (Not directly reflected in the link as some further changes were made to the ranking by the organizers) tied with another team. The best model turned out to be the one by UHH-LT.

--- 
### Directory Structure:
- ```Data```: Stores Data
- ```Data Preprocessing```: Contains the Preprocessing Files
- ```EDA```: Our Visualizations and Analysis
- ```Models```: Contains code for various models.
- ```Results```: Stores resulting labels for each model.
- ```Word Embeddings```: Contains code for producing embedded vectors of tweets.

