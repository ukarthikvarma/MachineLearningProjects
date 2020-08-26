import pickle

import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('saved_models/best_model_without_reviews/rfBestWithoutReviews.pkl', 'rb'))
cv_model = pickle.load(open('saved_models/cv_model/cv.pkl', 'rb'))
lb_model = pickle.load(open('saved_models/lightbgm_model/lb_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/predict',methods=['POST'])
# def predict():
#   path = 'data/input1.csv'
#  data_obj = Preprocessing(path)
# df,X_df,y_df,X_df_train,X_df_test,y_df_train,y_df_test=data_obj.preprocess_with_reviews()
# model_obj = ModelTraining(X_df_train.values,X_df_test.values,y_df_train.values,y_df_test.values,'df1')
# model_obj.int_training()
# model_obj.hyperparameter_tuning()
# return 'Preprocessed and Model Trained'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        top_city = ['London', 'Paris', 'Milan', 'Madrid', 'Rome']
        top_cuisine = ['Pizza', 'European', 'Italian', 'Spanish', 'French']
        x = [x for x in request.form.values()]
        features = {
            'Ranking': int(x[0]),
            'Price Range': int(x[1]),
            'Number of Reviews': int(x[2]),
            'Rome': 0,
            'London': 0,
            'Madrid': 0,
            'Milan': 0,
            'Others_city': 0,
            'Paris': 0,
            'Pizza': 0,
            'European': 0,
            'French': 0,
            'Italian': 0,
            'Spanish': 0
        }
        if x[3] in top_city:
            features[x[3]] = 1
        else:
            features['Others_city'] = 1

        if x[4] in top_cuisine:
            features[x[4]] = 1

        features_updated = []
        for i in features.values():
            features_updated.append(i)
        final_features = [np.array(features_updated)]

        output1 = np.round(model.predict(final_features)[0], 1)
        review = x[5]
        data = [review]
        vect = cv_model.transform(data).toarray()
        output2 = np.round(lb_model.predict(vect)[0], 1)
        final_prediction = np.round((output1 + output2) / 2, 2)
        return render_template('result.html',
                               predicted_rating='The Predicted rating is : {}'.format(final_prediction))


if __name__ == '__main__':
    app.run(debug=True)
