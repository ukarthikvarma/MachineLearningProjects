import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from application_logger import Logger
from sklearn.metrics import mean_squared_error , mean_absolute_error
import os

def metrics(true,pred):
  print('MAE: {}'.format(mean_absolute_error(true,pred)))
  print('MSE: {}'.format(mean_squared_error(true,pred)))



stemmer = PorterStemmer()


class Review_preprocessing:
    def __init__(self, rating, reviews, length):
        self.rating = rating
        self.reviews = reviews
        self.length = length
        self.logger = Logger()
        if os.path.exists(
                'preprocessing_logs/reviews_preprocessing_logs.txt'):
            os.remove(
                'preprocessing_logs/reviews_preprocessing_logs.txt')
        self.file_object = open(
            'preprocessing_logs/reviews_preprocessing_logs.txt',
            'a+')

    def reviews_preprocessing(self):
        self.logger.log(self.file_object, 'Preprocessing of Reviews started')
        print('Preprocessing of Reviews started')
        self.corpus = []
        self.logger.log(self.file_object, 'Corpus variable is defined')
        print('Corpus variable is defined')
        self.logger.log(self.file_object, 'Cleaning of reviews is started')
        print('Cleaning of reviews is started')
        # cleaning of reviews started
        for i in range(0, self.length):
            self.review = re.sub("[^a-zA-Z0-9]+", ' ', self.reviews[i])
            self.review = re.sub("[1,2,3,4,5,6,7,8,9,0]",' ', self.review)
            self.review = self.review.lower()
            self.review = self.review.split()
            self.review = [stemmer.stem(self.word) for self.word in self.review if
                           self.word not in set(stopwords.words('english'))]
            self.review = ' '.join(self.review)
            self.corpus.append(self.review)
        self.logger.log(self.file_object, 'Cleaning of reviews is completed')
        print('Cleaning of reviews is completed')
        self.logger.log(self.file_object,
                        'Creating a Bag of Words using Countvectorizor techinque with max_features as 3000')
        print('Creating a Bag of Words using Countvectorizor techinque with max_features as 3000')
        self.cv = CountVectorizer(max_features=3000)
        self.X = self.cv.fit_transform(self.corpus).toarray()
        self.y = self.rating
        self.logger.log(self.file_object, 'Bag of words is created and Features and labels are seperated')
        print('Bag of words is created and Features and labels are seperated')

        self.logger.log(self.file_object, 'Creating a pickle file for Countvectoriser')
        self.file_cv = open('saved_models/cv_model/cv.pkl', 'wb')
        pickle.dump(self.cv, self.file_cv)
        self.file_cv.close()
        self.logger.log(self.file_object, 'Pickle file created for Countvectoriser')
        print('Pickle File created for CountVectorizer')

        self.file_object.close()

        return self.X, self.y

    def reviews_model_training(self, X, y):
        if os.path.exists(
                'model_training_logs/reviews_model_training_logs.txt'):
            os.remove(
                'model_training_logs/reviews_model_training_logs.txt')
        self.file_object = open(
            'model_training_logs/reviews_model_training_logs.txt',
            'a+')
        self.logger.log(self.file_object, 'Reviews model training started')
        self.logger.log(self.file_object, 'Starting to create training and testing sets')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=4)
        self.logger.log(self.file_object, 'Seperated training and testing sets')
        self.lb = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=40, learning_rate=0.05, n_estimators=1000)
        print('Model Training started')
        self.logger.log(self.file_object, 'Model Training started')
        self.lb.fit(self.X_train, self.y_train)
        self.logger.log(self.file_object, 'Model Training ended')
        print('Model Training ended')
        self.y_pred_lb = self.lb.predict(self.X_test)
        self.logger.log(self.file_object, 'Values predicted')
        self.logger.log(self.file_object,
                        'MSE value of model is {}'.format(mean_squared_error(self.y_test, self.y_pred_lb)))
        print('Scores of Model is : ')
        metrics(self.y_test, self.y_pred_lb)

        self.logger.log(self.file_object, 'Saving Model as a pickle file')
        self.file_lb = open('saved_models/lightbgm_model/lb_model.pkl', 'wb')
        pickle.dump(self.lb, self.file_lb)
        self.file_lb.close()
        print('Model saved as pickle file')
        self.logger.log(self.file_object, 'Model Saved as a pickle file')
        self.logger.log(self.file_object, 'Exiting the model_training_class')
        print('Model training is completed')
        self.file_object.close()