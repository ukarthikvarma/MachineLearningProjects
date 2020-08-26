import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import warnings

warnings.filterwarnings('ignore')
from application_logger import Logger
from reviews_preprocessing import Review_preprocessing


class Preprocessing:
    def __init__(self, path):
        self.path = path
        self.logger = Logger()
        if os.path.exists('preprocessings_logs/preprocessing_logs.txt'):
            os.remove('preprocessings_logs/preprocessing_logs.txt')
        self.file_object = open('preprocessings_logs/preprocessing_logs.txt', 'a+')

    def preprocess(self):
        self.logger.log(self.file_object, 'Reading Ratings reviews dataset')
        self.dataframe = pd.read_csv(self.path, index_col=0)
        self.dataframe.reset_index(inplace=True)
        self.logger.log(self.file_object, 'Sucessfully read the dataframe the path sepicified....')
        print('Sucessfully read dataframe')
        self.logger.log(self.file_object, '****PREPROCESSING STARTED FOR DATAFRAME WITHOUT REVIEWS****')

        self.logger.log(self.file_object, 'Converting the [[],[]] values into null values of Reviews column')
        for i in range(len(self.dataframe)):
            if self.dataframe['Reviews'][i] == '[[], []]':
                self.dataframe['Reviews'][i] = np.nan
        self.logger.log(self.file_object, 'Converted [[], []] into null values')

        self.logger.log(self.file_object, 'Dropping all the null values in df')
        self.dataframe.dropna(inplace=True)
        self.dataframe.reset_index(inplace=True, drop=True)
        self.logger.log(self.file_object, 'Dropped all the null values in the dataset....')

        self.logger.log(self.file_object, 'Dropping Name column')
        self.dataframe.drop('Name', axis=1, inplace=True)
        self.logger.log(self.file_object, 'Sucessfully Dropped Name column')

        self.logger.log(self.file_object, 'Starting to group the categorical values')
        self.top_6_city = self.dataframe['City'].value_counts()[0:6].to_dict()
        self.top_5_cuisines = self.dataframe['Cuisine Style'].value_counts()[0:5].to_dict()
        self.logger.log(self.file_object,
                        'Found Top 6 six cities and top 5 cuisine style , we will group remaining into Other cities and other cuisine style Categories')
        for i in range(len(self.dataframe)):
            if self.dataframe['City'][i] not in self.top_6_city.keys():
                self.dataframe['City'][i] = 'Other_cities'
        self.logger.log(self.file_object, 'Sucessfully grouped cities and other cities')
        for i in range(len(self.dataframe)):
            if self.dataframe['Cuisine Style'][i] not in self.top_5_cuisines.keys():
                self.dataframe['Cuisine Style'][i] = 'Other_cuisine'
        self.logger.log(self.file_object, 'Sucessfuly grouped Cuisine Styles and Other cuisine')

        self.logger.log(self.file_object, 'Starting to ordinal codin the price range columns')
        self.dataframe['Price Range'] = self.dataframe['Price Range'].replace({'$': 0, '$$ - $$$': 1, '$$$$': 2})
        self.logger.log(self.file_object, 'Sucessfully encoded price range column')

        self.length = len(self.dataframe)
        self.logger.log(self.file_object, 'Calling Reviews Preprocessing class for training of reviews')
        self.review_obj = Review_preprocessing(self.dataframe['Rating'], self.dataframe['Reviews'], self.length)
        self.logger.log(self.file_object, 'Reviews Preprocessing started')
        self.X, self.y = self.review_obj.reviews_preprocessing()
        self.logger.log(self.file_object, 'Reviews Prerocessing completed')
        self.logger.log(self.file_object, 'Reviews model training started')
        self.review_obj.reviews_model_training(self.X, self.y)
        self.logger.log(self.file_object, 'Reviews model Trainig ended Models saved as pickle files')
        self.logger.log(self.file_object, 'Exiting Reviews Preprocessing class')

        self.logger.log(self.file_object, 'Dropping Reviews Column')
        self.dataframe.drop('Reviews', axis=1, inplace=True)
        self.logger.log(self.file_object, 'Droppe Reviews column')

        self.logger.log(self.file_object, 'Starting to Dummy coding the object type variables')
        self.dataframe = pd.get_dummies(self.dataframe, prefix_sep='_', drop_first=True)
        self.logger.log(self.file_object, 'Sucessfully dummy encoded object variables')

        self.logger.log(self.file_object, 'Shuffling the Dataframe')
        self.dataframe = shuffle(self.dataframe)
        self.dataframe.reset_index(inplace=True, drop=True)
        self.logger.log(self.file_object, 'Sucessfully Shuffled Dataframe')

        self.logger.log(self.file_object, 'Seprating Features and labels')
        self.X_df = self.dataframe.drop('Rating', axis=1)
        self.y_df = self.dataframe['Rating']
        self.logger.log(self.file_object, 'Seperated Features and labels')

        self.logger.log(self.file_object, 'Splitting data into training and testing data')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_df, self.y_df,
                                                                                test_size=0.3,
                                                                                random_state=4)
        self.logger.log(self.file_object, 'Sucessfully Seperated Training and testing data')

        self.logger.log(self.file_object, '****Sucessfully Completed Preprocessing step for DF1****')
        self.file_object.close()

        print('Preprocessing of DF1 is completed sucessfully')

        return self.dataframe, self.X_df, self.y_df, self.X_train, self.X_test, self.y_train, self.y_test