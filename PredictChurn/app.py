
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

df_1 = pd.read_csv("first_telc.csv")

q = ""


@app.route('/', methods=['GET'])
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    '''
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    '''

    inputQuery1 = request.form['citizen']
    inputQuery2 = request.form['monthly-charges']
    inputQuery3 = request.form['total-charges']
    inputQuery4 = request.form['gender']
    inputQuery5 = request.form['partner']
    inputQuery6 = request.form['dependents']
    inputQuery7 = request.form['phone-service']
    inputQuery8 = request.form['multiple-lines']
    inputQuery9 = request.form['internet-service']
    inputQuery10 = request.form['online-security']
    inputQuery11 = request.form['online-backup']
    inputQuery12 = request.form['device-protect']
    inputQuery13 = request.form['tect-support']
    inputQuery14 = request.form['streaming-tv']
    inputQuery15 = request.form['streaming-movies']
    inputQuery16 = request.form['contract']
    inputQuery17 = request.form['paperless-billing']
    inputQuery18 = request.form['payment-method']
    inputQuery19 = request.form['tenure']

    model = pickle.load(open("model.sav", "rb"))

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7,
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    df_2 = pd.concat([df_1.iloc[:, 1:], new_df], ignore_index=True)
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(
    	int), range(1, 80, 12), right=False, labels=labels)
    # drop column customerID and tenure
    df_2.drop(columns=['tenure'], axis=1, inplace=True)

    # dummified_col = ['gender','SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    #                  'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    #                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    #                  'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']

    # excluded_col = ['MonthlyCharges', 'TotalCharges']

    # new_df__dummies = pd.get_dummies(df_2.drop(columns=excluded_col),
    #                                  columns=dummified_col,
    #                                  drop_first=True)

    # new_df__dummies()

# Insert the excluded columns at the beginning of the dummified dataframe
    # for column in excluded_col[::-1]:
    #     new_df__dummies.insert(0, column, df_2[column])

    new_df__dummies = pd.get_dummies(df_2, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                           'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'])

    # new_df__dummies[['SeniorCitizen','MonthlyCharges','TotalCharges']] = df_2[['SeniorCitizen','MonthlyCharges','TotalCharges']]

    print(df_1.columns, 'df1 columns')
    print(df_2.columns, 'df_2 col')
    print(new_df.columns, 'new_df')
    print(new_df__dummies.tail(1).columns, 'columns')
    print(new_df__dummies.tail(1), 'new_df_dum')

    # final_df=pd.concat([new_df__dummies, new_dummy], axis=1)

    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:, 1]

    if single == 1:
        o1 = "is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)

    return render_template('output.html', output1=o1, output2=o2,
                           firstName=request.form['firstName'],
                           lastName=request.form['lastName'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
