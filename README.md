# Project 3 - Subreddit Classification with Natural Language Processing

*Author: Grace Campbell*


## Problem Statement

[Reddit](https://reddit.com) is a content aggregation website where members can submit links, text posts, images, and videos, which other members can then comment on and discuss. The posts "are organized by subject into user-created boards called 'subreddits', which cover a variety of topics including news, science, movies, video games, music, books, fitness, food, and image-sharing." ([Wikipedia](https://en.wikipedia.org/wiki/Reddit))

There are two subreddits I am interested in: /r/News and /r/TheOnion. The first contains titles of news articles, while the second contains titles of satirical news articles. Can I build a classification model using natural language processing that can accurately predict which subreddit a given post came from?

#### Project Directory
1. Data Preparation 
    - [Data Gathering](http://localhost:8889/notebooks/projects/project_3/data-gathering.ipynb)
    - [Exploratory Data Analysis](http://localhost:8889/notebooks/projects/project_3/exploratory-data-analysis.ipynb)
2. Modeling
    - [Naive Bayes](http://localhost:8889/notebooks/projects/project_3/modeling-naive-bayes.ipynb)
    - [*k*-Nearest Neighbors](http://localhost:8889/notebooks/projects/project_3/modeling-knn.ipynb)
    - [Support-Vector Machine](http://localhost:8889/notebooks/projects/project_3/modeling-svm.ipynb)
    - [Final Testing on New Data](http://localhost:8889/notebooks/projects/project_3/final-models-testing.ipynb)
    
    
## Data Exploration

After scraping Reddit's API for the subreddits in question, I created a table of each title with its class. 

In this case, 1 = /r/TheOnion, and  0 = /r/News.
![image](https://github.com/GraceCampbell/Fake-News-Classification-NLP/blob/master/materials/data.PNG)
___
![image](https://github.com/GraceCampbell/Fake-News-Classification-NLP/blob/master/materials/fig1.png)
___
![image](https://github.com/GraceCampbell/Fake-News-Classification-NLP/blob/master/materials/fig2.png)
___
![image](https://github.com/GraceCampbell/Fake-News-Classification-NLP/blob/master/materials/table.PNG)

## Modeling

I chose to use Naive Bayes, $k$-Nearest Neighbors, and Support-Vector Machines to model this problem.

![image](https://github.com/GraceCampbell/Fake-News-Classification-NLP/blob/master/materials/metrics.PNG)

Naive Bayes and SVM both had high accuracy, sensitivity, and specificity, while $k$-Nearest Neighbors had high accuracy and high sensitivity, but low specificity. In a real-world application, it is equally important to me that this model be able to correctly predict when a post is satirical and when it is real. The positive class in this case (/r/TheOnion) does not hold more weight than the negative class (/r/News), so I would rather the model be very accurate than very sensitive or specific.

That being said, $k$-Nearest Neighbors, at least not this iteration, is not the best model due to its low specificity compared to the other two models. Naive Bayes and SVM are both good models that I would use for this classification problem. To improve all three of these models, I would like to gather more data.
