# Fake News Predictor

## Project Overview

This project aims to build a Fake News Predictor using various machine learning techniques. The model is trained on datasets of news articles, labeled as fake or real, and includes preprocessing steps such as text cleaning, tokenisation, and vectorisation.

## Data

This project uses the **WELFake** dataset, which is designed for fake news classification. The dataset consists of 72,134 news articles, with 35,028 labeled as real news and 37,106 labeled as fake news. The data was sourced from four popular news datasets—Kaggle, McIntire, Reuters, and BuzzFeed Political—to provide a diverse set of text data and to prevent overfitting in machine learning models.

### Dataset Structure

The dataset contains the following columns:

- **Serial number**: An integer identifier for each news article (starting from 0).
- **Title**: The title or headline of the news article.
- **Text**: The main content of the news article.
- **Label**: Indicates whether the article is real or fake:
  - `0`: Fake news
  - `1`: Real news

The dataset provides a robust foundation for training and evaluating machine learning models aimed at detecting fake news. The labels are balanced with a nearly equal number of real and fake news articles, making it suitable for binary classification tasks.

### Source

The dataset was published in the following paper:

- **IEEE Transactions on Computational Social Systems**: pp. 1-13 (doi: [10.1109/TCSS.2021.3068519](https://doi.org/10.1109/TCSS.2021.3068519)).

You can access the dataset on Kaggle via this [link](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).


## Installation

### Setting Up the Environment

1. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   conda create --name fake-news-predictor python=3.10
   conda activate fake-news-predictor
2. **Install Required Packages**
    - Install all the dependencies using the requirements.txt file:
    ```bash
    pip install -r requirements.txt
    