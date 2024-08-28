# Fake News Predictor

### Collaborators: 
Sindhuja Sirigeri, Tammy Powell, Wendy Ware, Uthpalie Thilakaratna-Attygalle

----------------------------------------------------------------------------

## Project Overview

This project aims to build a Fake News Predictor using various machine learning techniques. The model is trained on a dataset of news articles labeled as fake or real and includes preprocessing steps such as text cleaning, tokenisation, and vectorisation. The project explores multiple models, including Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).

## App Link
App Link : https://fake-vs-fact-predictor.streamlit.app/

## Data

### Dataset

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

## Repository Structure
- /data/splits/: Contains the train_test_split.h5 file.
- /models/: Stores all trained model files (.pkl) and corresponding metrics (.json).
- /notebooks/: Jupyter notebooks detailing the preprocessing, model training, and evaluation steps.
- README.md: Overview of the project, including a step-by-step process.

## Conclusion
The project successfully trained and optimised multiple models for fake news detection, ready for integration into a Streamlit app. Detailed documentation and best practices ensure that the models can be deployed efficiently and effectively.


-------------------------------------------------------------------------------------------------


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

## Project Process

1. **Data & Methodology Research**
Research was conducted to find a suitable dataset for the project. The WELFake dataset was selected due to its balanced representation of real and fake news articles.

2. **Data Preprocessing**
Preprocessing involved:
- Dropping unnecessary columns.
- Handling missing values.
- Removing stopwords, numerics, and special characters.
- Lemmatisation to reduce words to their base form.
- Case folding to ensure uniformity.

3. **Tokenisation and Padding**
The text was tokenised and padded to ensure consistent sequence lengths, necessary for feeding the data into machine learning models.

4. **Exploratory Data Analysis (EDA)**
EDA involved:
- **Class Distribution Analysis**: Understanding the balance between fake and real news.
- **Text Length Distribution**: Analysing the distribution of text lengths.
- **Word Cloud**: Visualising the most common words.
- **Correlation Analysis**: Exploring the relationship between text length and labels.
- **Handling Skewness**: Addressed the skewness in the data with strategies like truncation, balancing, and robust models.

5. **Vectorisation**
Experimented with Word2Vec and TF-IDF techniques, with TF-IDF being selected for the final implementation.

6. **Feature Engineering**
Combined the vectors and inspected the matrix before splitting the data into training and testing sets.

7. **Model Selection and Modelling**
The following models were explored:
- **Logistic Regression**: Optimized using GridSearchCV.
- **Decision Tree**: Tuned for specific hyperparameters.
- **Random Forest**: Applied Incremental PCA (IPCA) for dimensionality reduction.
- **Support Vector Machine (SVM)**: Detailed recommendations were provided for handling its computational intensity.
- **K-Nearest Neighbors (KNN)**: Implemented with optimized hyperparameters.

8. **Model Optimisation**

9. **Deployment on Streamlit**
The project is prepared for deployment on Streamlit, with considerations for handling large datasets and real-time user input preprocessing.

10. **Project Reporting & Presentation**
Final reporting includes a detailed README.md file and a presentation summarising the project process, findings, and outcomes.


### Data & Methodology Research 

At the begining of the project, research was conducted to find datasets suitable for the project. The criteria included, selecting datasets with enough data records and datasets that has a near 50/50 samples of both fake and real data records. 

### Data Preprocessing

This was identified as a crucial step as the dataset required cleaning and processing prior to being transformed for data modeling. The following were followed:
* Dropping unnecessary columns
* Handling missing values
* Removing stopwords, numerics, special characters etc.
* Lemmitization (Lemmatisation helps reduce words to their base form (e.g., "running" to "run"). NLTK’s WordNetLemmatizer is used for this.)
* Case Folding (Lowercasing)

### Tokenization and Padding

The next step was to tokenise the text and pad the sequences. Tokenisation converts text into sequences of integers where each integer corresponds to a specific word in the text. Padding ensures that all sequences have the same length, which is necessary for feeding the data into most machine learning models.

![cumalitive](https://github.com/user-attachments/assets/365f489b-1e0c-42ea-8149-f375f6c694b1)


![distribution](https://github.com/user-attachments/assets/37d6c5ca-d382-4fed-b089-2cf585e20ac4)


### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is the process of analysing and investigating the dataset to understand its characteristics better, most of the time, visualizing. This is beforehand so that we can use the dataset for our final purpose, in this case for machine learning modeling.

1. Class Distribution Analysis

![class distribution analysis](https://github.com/user-attachments/assets/61d9c427-ed97-4cb4-b747-9e14ffdc8e2f)



2. Text Length Distribution

![text length distribution](https://github.com/user-attachments/assets/5662ea7e-3a1c-4dda-8902-129dff236055)

Given the skewness of the data, particularly in the distribution of text lengths, there are a few strategies one might consider to improve one's model outcomes. Here are some recommendations:

* Truncate Long Sequences: Extremely long sequences may introduce noise or overwhelm your model, especially if they are outliers. Consider truncating sequences that are longer than a certain threshold (e.g., the 95th or 99th percentile of text length). You've already padded sequences to a maximum length, so this step might involve ensuring that your max sequence length is appropriate.
* Handling Short Sequences: Very short sequences might not provide enough information for the model to make accurate predictions. You can consider filtering out sequences that are too short, or alternatively, use techniques like padding to handle them appropriately.
* Apply Data Augmentation: If the skewness is related to a class imbalance (e.g., more real news than fake news), you might want to apply data augmentation techniques. This could include oversampling the minority class or using synthetic data generation techniques like SMOTE.
* Normalisation or Standardisation: For numerical features (e.g., text_length), normalisation or standardisation can help mitigate the impact of skewness. However, since you are working primarily with text data, this might be more applicable to secondary features you might engineer from the text.
* Use Robust Models: Some models are more robust to skewed data than others. For example, tree-based models like Random Forest or Gradient Boosting can handle skewed data better. Alternatively, you could explore models that specifically account for skewness.
* Balancing the Dataset: If the skewness is related to an imbalance in the labels (e.g., more fake news than real news), you can balance the dataset by undersampling the majority class or oversampling the minority class. However, this needs to be done carefully to avoid overfitting, especially when oversampling.
* Use Log Transformation: If the skewness is extreme, consider applying a log transformation to reduce the impact of outliers. However, this is generally more applicable to continuous numerical features rather than text lengths.

  
3. Word Cloud

![Word Cloud](https://github.com/user-attachments/assets/20254260-9efe-4c3d-b21b-3ed58f784220)


4. Most Common Words

The following were found as the most common fake and real words.

Most common words in fake news: [('said', 184625), ('trump', 90859), ('mr', 66097), ('would', 62656), ('president', 51066), ('new', 50062), ('us', 48582), ('state', 47122), ('year', 46200), ('one', 43608)]
Most common words in real news: [('trump', 105582), ('people', 48821), ('said', 47143), ('one', 46036), ('would', 43146), ('clinton', 41855), ('president', 37794), ('us', 36470), ('hillary', 32281), ('like', 32276)]


5. Check for Duplicate Entries

8456 rows were identified as duplicate entries.
   
6. Correlation Analysis

![Text length vs Labels](https://github.com/user-attachments/assets/17ef2f23-8a26-4cc8-81b0-687e89ae5637)


7. Visualizing Data Distribution

![Text Length Distibution by Label](https://github.com/user-attachments/assets/6ac75dcf-858d-483c-80d3-d36df7e510cb)


8. Transformation

![Transformation](https://github.com/user-attachments/assets/c4db8ff8-8ec7-4034-a453-220770af0b41)



### Vectorization

Vectorization is the process in which you convert raw data, in this case text, in to vectors of real numbers so that it produce a feature extraction so that a machine learning model can be trained.

In this project we experiemented with Word2Vec, pre-trained model by Google and TF-IDF technique which is a refinement over the simple bag-of-words model, by allowing the weight of words to depend on the rest of the corpus. For the implementation TF-IDF technique was selected.


### Feature Engineering

In this step we have combined the vectors and inspected the combined matrix, before splitting the data into train and test sets.

![Comparision of log_text_length Distribution](https://github.com/user-attachments/assets/4fa89948-e2d8-4c78-a5da-2e4c6b3fde9a)


### Model Selection and Modelling

Research was conducted initially, to understand which modeling technique would be ideal for this purpose. Logistical Regression Model, Decision Tree Model, KNN, SVM and Random Forest models were fitting  with the dataset. The best performing model was Logistical Regression Model at 94%, hence was selected for the final optimization and deployment.

![Accuracy Comparison ](https://github.com/user-attachments/assets/caa08888-6a96-4e4b-8389-a8851ed876af)


Further research was included to be used for enhancing this solution for furture. <Include recommendations here>



## Model Optimization

Optimisation was performed through hyperparameter tuning and dimensionality reduction:
- **GridSearchCV**: Used for finding the best parameters.
- **Incremental PCA**: Applied to reduce dimensions while retaining variance.


## Deployment

For deployment of the model and for user interactions with the model Streamlit.io (https://streamlit.io/) was used.









    
    
