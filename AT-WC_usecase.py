#!/usr/bin/env python
# coding: utf-8

# **Simple Linear regression**
# Simple linear regression is a regression model that estimates the relationship between one independent variable and a dependent variable using a straight line.

# **Problem Statement**
# 
# The Waist Circumference – Adipose Tissue Relationship:
# 
# Studies have shown that individuals with excess Adipose tissue (AT) in their abdominal region have a higher risk of cardio-vascular diseases.
# To assess the health conditions of a patient, doctor must get a report on the patients AT values. Computed Tomography, commonly called the CT Scan is the only technique that allows for the precise and reliable measurement of the AT (at any site in the body). 
# 
# The problems with using the CT scan are:
# - Many physicians do not have access to this technology
# - Irradiation of the patient (suppresses the immune system)
# - Expensive
# 
# The Hospital/Organization wants to find an alternative solution for this problem, which can allow doctors to help their patients efficiently.
# 
# 

# **CRISP-ML(Q) process model describes six phases:**
# 
# - Business and Data Understanding
# - Data Preparation (Data Engineering)
# - Model Building (Machine Learning)
# - Model Evaluation and Tunning
# - Deployment
# - Monitoring and Maintenance
# 

# ![Picture1.jpg](attachment:Picture1.jpg)

# **Objective(s):** Minimize the risk for patients
# or 
# Maximize the convience to doctors in assisting their patients
# 
# **Constraints:** CT Scan is the only option
# 
# **Research:** A group of researchers conducted a study with the aim of predicting abdominal AT area using simple anthropometric measurements, i.e., measurements on the human body
# 
# 
# **Proposed Plan:**
# The Waist Circumference – Adipose Tissue data is a part of this study wherein the aim is to study how well waist circumference (WC) predicts the AT area
# 
# 
# **Benefits:**
# Is there a simpler yet reasonably accurate way to predict the AT area? i.e.,
# - Easily available
# - Risk free
# - Inexpensive
# 

# **Data Collection**

# Data: 
#     AT values from the historical Data
#     Waist Circumference of these patients.
# 
# Collection:
# 1. Evaluate the available Hospital records for relevant data (CT scan of patients)
# 
# 2. Record the Waist Circumference of patients - Primary Data
# 
# - Strategy to Collection Primary Data:
#     Call out the most recent patients (1 year old) with an offer of free consultation from a senior doctor to attract them to visit hospital.
#     Once the paitents visit the hospital, we can record their Waist Circumference with accuracy.

# # Explore the Patients Database (MySQL)

# Connect to the MySQL DB source for Primary data




# MySQL DB 
# pip install mysql-connector-python

import mysql.connector as sql

mydb = sql.connect(
  host="localhost",
  user="dengg1",
  passwd="dengg1",
  auth_plugin='mysql_native_password'
)


print(mydb)

mycursor = mydb.cursor()




# Define SQL query
dblist = "SHOW DATABASES"

# Execute the defined query
mycursor.execute(dblist)

# Fetch the output from the query
dblist_results = mycursor.fetchall()

# print the results in Python console
for x in dblist_results:
    print(x)




# Create Table in Schema Cardio
import mysql.connector as sql

mydb = sql.connect(
  host="localhost",
  user="dengg1",
  passwd="dengg1",
  database = 'cardio',
  auth_plugin='mysql_native_password'
)

print(mydb)

mycursor = mydb.cursor()




# Define SQL query
tblist = "SHOW TABLES"

# Execute the defined query
mycursor.execute(tblist)

# Fetch the output from the query
tblist_results = mycursor.fetchall()

# print the results in Python console
for x in tblist_results:
    print(x)


# Read the data from the tables to Analyse the features
SQL_where = "describe patients"

mycursor.execute(SQL_where)

SQL_where_results = mycursor.fetchall()

for x in SQL_where_results:
    print(x)


# Read the data from the tables to Analyse the features
SQL_where = "SELECT * from patients"

mycursor.execute(SQL_where)

SQL_where_results = mycursor.fetchall()

for x in SQL_where_results:
    print(x)


# Read the data from the tables to Analyse the features
SQL_where = "describe waist"

mycursor.execute(SQL_where)

SQL_where_results = mycursor.fetchall()

for x in SQL_where_results:
    print(x)


# Read the data from the tables to Analyse the features
SQL_where = "SELECT * from waist"

mycursor.execute(SQL_where)

SQL_where_results = mycursor.fetchall()

for x in SQL_where_results:
    print(x)


# Import only the required features into Python for Processing


SQL_join = "SELECT A.Patient, A.AdiposeTissue, A.Sex, A.Age, B.waist             from patients as A             Inner join waist as B             on A.Patient = B.Patient"
            
mycursor.execute(SQL_join)

SQL_results = mycursor.fetchall()

for x in SQL_results:
    print(x)


# Importing necessary libraries
import pandas as pd # deals with data frame        # for Data Manipulation"
import numpy as np  # deals with numerical values  # for Mathematical calculations"


wcat_full = pd.DataFrame(SQL_results)

wcat_full.info()


wcat_full.columns = 'id', 'AT', 'Sex', 'Age', 'Waist'

wcat_full.info()


wcat_full.describe()


wcat_full.Sex.value_counts()


# # Load the Data and perform EDA and Data Preprocessing

# ############ optional ###############


# Importing necessary libraries
import pandas as pd # deals with data frame        # for Data Manipulation"
import numpy as np  # deals with numerical values  # for Mathematical calculations"

wcat = pd.read_csv(r"C:\Data\wc-at.csv")

wcat.info()


# #########################


wcat = wcat_full.drop(["id", "Sex", "Age"], axis = 1)


#### Descriptive Statistics and Data Distribution

wcat.describe()


# Graphical Representation
import matplotlib.pyplot as plt         # mostly used for visualization purposes 

plt.bar(height = wcat.AT, x = np.arange(1, 110, 1))


plt.hist(wcat.AT) #histogram


plt.boxplot(wcat.AT) #boxplot


plt.bar(height = wcat.Waist, x = np.arange(1, 110, 1))


plt.hist(wcat.Waist)


plt.boxplot(wcat.Waist)


# The above are manual approach to perform Exploratory Data Analysis (EDA). The alternate approach is to Automate the EDA process using Python libraries.
# 
# Auto EDA libraries:
# - Sweetviz
# - dtale
# - pandas profiling
# - autoviz

# 
# # **Automating EDA with Sweetviz:**
# 

# Using sweetviz to automate EDA is pretty simple and straight forward. 3 simple steps will provide a detailed report in html page.
# 
# step 1. Install sweetviz package using pip.
# - !pip install sweetviz
# 
# step2. import sweetviz package and call analyze function on the dataframe.
# 
# step3. Display the report on a html page created in the working directory with show_html function.
# 


import sweetviz as sv

# Analyzing the dataset
report = sv.analyze(wcat)

# Display the report
report.show_notebook()  # integrated report in notebook

# report.show_html('EDAreport.html') # html report generated in working directory


# # Bivariate Analysis
# Scatter plot
plt.scatter(x = wcat['Waist'], y = wcat['AT']) 


## Measure the strength of the relationship between two variables using Correlation coefficient.

np.corrcoef(wcat.Waist, wcat.AT) 


# Covariance
cov_output = np.cov(wcat.Waist, wcat.AT)[0, 1]
cov_output

# wcat.cov()


import seaborn as sb

dataplot = sb.heatmap(wcat.corr(), annot=True, cmap="YlGnBu")


# # Linear Regression using statsmodels package

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('AT ~ Waist', data = wcat).fit()


model.summary()


pred1 = model.predict(pd.DataFrame(wcat['Waist']))

pred1


# Regression Line
plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


# Error calculation (error = AV - PV)
res1 = wcat.AT - pred1

res1


print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(wcat['Waist']), y = wcat['AT'], color = 'brown')
np.corrcoef(np.log(wcat.Waist), wcat.AT) #correlation


model2 = smf.ols('AT ~ np.log(Waist)', data = wcat).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(wcat['Waist']))

# Regression Line
plt.scatter(np.log(wcat.Waist), wcat.AT)
plt.plot(np.log(wcat.Waist), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res2 = wcat.AT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = wcat['Waist'], y = np.log(wcat['AT']), color = 'orange')
np.corrcoef(wcat.Waist, np.log(wcat.AT)) #correlation


model3 = smf.ols('np.log(AT) ~ Waist', data = wcat).fit()
model3.summary()


pred3 = model3.predict(pd.DataFrame(wcat['Waist']))

# Regression Line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()


pred3


pred3_at = np.exp(pred3)
print(pred3_at)

# Error calculation
res3 = wcat.AT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation 
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = wcat).fit()
model4.summary()


pred4 = model4.predict(pd.DataFrame(wcat))
print(pred4)
print('\n')
# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = wcat.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
print(X_poly)


plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = wcat.AT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


# # Evaluate the best model


from sklearn.model_selection import train_test_split

train, test = train_test_split(wcat, test_size = 0.2)

plt.scatter(train.Waist, np.log(train.AT))

plt.figure(2)
plt.scatter(test.Waist, np.log(test.AT))


# Fit the best model on train data
finalmodel = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = train).fit()
finalmodel.summary()


# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT


# Model Evaluation on Test data
test_res = test.AT - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)

test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT


# Model Evaluation on train data
train_res = train.AT - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)

train_rmse


# # Deploy the Best Model using Flask

from flask import Flask, render_template, request  #, url_for
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# import pickle
# from sklearn.linear_model import LinearRegression
# from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])

def predict():
    df = pd.read_csv(r"C:\Data\wc-at.csv")

    regressor = smf.ols('np.log(AT) ~ Waist + I(Waist * Waist)', data = df).fit()

    
    if request.method == 'POST':
        value = request.form['val']
        pred = pd.DataFrame([value])
        pred.columns = ['Waist']
        my_pred = regressor.predict(pred.astype('float32'))
        AT_pred = np.exp(my_pred)
    return render_template('result.html', prediction = AT_pred)


if __name__ == '__main__':
    app.run(debug = False)





