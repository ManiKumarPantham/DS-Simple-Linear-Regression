##########################################################################################################
Problem Statement: A certain food-based company conducted a survey with the help of a fitness company to find 
                the relationship between a person’s weight gain and the number of calories they consumed 
                to come up with diet plans for these individuals. Build a Simple Linear Regression model 
                with calories consumed as the target variable. Apply necessary transformations and 
                record the RMSE and correlation coefficient values for different models. 
#########################################################################################################

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as sfa
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Reading the dataset into Python
data = pd.read_csv(r"D:/Hands on/23_Simple Linear Regression/Assignment/calories_consumed.csv")

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Columns of the dataset
data.columns

# Renaming the dataset
data.rename({'Weight gained (grams)' : 'weight', 
             'Calories Consumed' : 'consumed'}, axis = 1, inplace = True)

# Columns of the dataset
data.columns

# Barplot
plt.bar(data.weight, height = np.arange(1, 15))
plt.bar(data.consumed, height = np.arange(1, 15))

#Boxplot
plt.boxplot(data.consumed)
plt.boxplot(data.weight)

# Density plot
sns.kdeplot(data.consumed)
sns.kdeplot(data.weight)

# Scatter plot
plt.scatter(data.weight, data.consumed)

# Correlation coefficient
data.corr()
np.corrcoef(data.weight, data.consumed)

# Calculating sum of duplicates
data.duplicated().sum()

# Checking the null values
data.isnull().sum()
data.isna().sum()


# Building a model
model1 = sfa.ols('consumed ~ weight', data = data).fit()

# Summary of the model
model1.summary()

# Prediction 
pred1 = model1.predict(pd.DataFrame(data['weight']))

# Scatter plot with predicted line
plt.scatter(data.weight, data.consumed)
plt.plot(data.weight, pred1, 'r')
plt.legend(['Obeseved values', 'Predicted values'])
plt.show()

# RMSE
err1 = data.consumed - pred1
serr1 = err1 * err1
mserr1 = np.mean(serr1)
rmse1 = np.sqrt(mserr1)
rmse1

# Scatter plot
plt.scatter(np.log(data.weight), data.consumed)

# Correlation coefficient
np.corrcoef(np.log(data.weight), data.consumed)

# Model
model2 = sfa.ols('consumed ~ np.log(weight)', data = data).fit()

# Summary
model2.summary()

# prediction
pred2 = model2.predict(pd.DataFrame(data['weight']))

# Scatter plot with predicted line
plt.scatter(np.log(data.weight), data.consumed)
plt.plot(np.log(data.weight), pred2, 'r')
plt.legend(['Obeseved values', 'Predicted values'])
plt.show()

# RMSE
err2 = data.consumed - pred2
serr2 = err2 * err2
mserr2 = np.mean(serr2)
rmsr2 = np.sqrt(mserr2)
rmsr2

# Scatter plot
plt.scatter(data.weight, np.log(data.consumed))

# Correlation coefficient
np.corrcoef(data.weight, np.log(data.consumed))

# Model builind
model3 = sfa.ols('np.log(consumed) ~ weight', data = data).fit()

# Summary of the model
model3.summary()

# Prediction
pred3 = model3.predict(pd.DataFrame(data['weight']))
pred3_exp = np.exp(pred3)

# Scatter plot with predicted line
plt.scatter(data.weight, np.log(data.consumed))
plt.plot(data.weight, pred3, 'r')
plt.legend(['Obeseved values', 'Predicted values'])
plt.show()

# RMSE
err3 = data.consumed - pred3_exp
serr3 = err3 * err3
mserr3 = np.mean(serr3)
rmse3 = np.sqrt(mserr3)
rmse3

# Model
model4 = sfa.ols('np.log(consumed) ~ weight + I(weight * weight) + I (weight * weight * weight)', data = data).fit()

# Model Summary
model4.summary()

# Prediction
pred4 = model4.predict(pd.DataFrame(data['weight']))
pred4_exp = np.exp(pred4)

# Regression line
poly_reg = PolynomialFeatures(degree = 3)
X = data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
print(X_poly)

# Scatter plot
plt.scatter(data.weight, np.log(data.consumed))
plt.plot(data.weight, pred4, 'r')
plt.legend(['Obeseved values', 'Predicted values'])
plt.show()

# RMSE
err4 = data.consumed - pred4_exp
serr4 = err4 * err4
mserr4 = np.mean(serr4)
rmse4 = np.sqrt(mserr4)
rmse4 

# Creating a dictionary with rmse values
data1 = pd.DataFrame({'Model' : pd.Series(["SLR", "Log model", "Exp model", "Poly model"]),
        "RMSE" : pd.Series([rmse1, rmsr2, rmse3, rmse4])})


# spliting the data into train_test split
train, test = train_test_split(data, test_size = 0.2, random_state = 0)

# Building a model
final_model = sfa.ols('consumed ~ weight', data = train).fit()

# Model summary
final_model.summary()

# prediction on test data
test_pred = final_model.predict(pd.DataFrame(test))

# Test RMSE
test_err = test.consumed - test_pred 
stest_err = test_err * test_err
mtest_err = np.mean(stest_err)
rmsetest_err = np.sqrt(mtest_err)
rmsetest_err

# Prediction on train data
train_pred = final_model.predict(pd.DataFrame(train))

# Train RMSE
train_err = train.consumed - train_pred 
strain_err = train_err * train_err
mtrain_err = np.mean(strain_err)
rmsetrain_err = np.sqrt(mtrain_err)
rmsetrain_err

########################################################################################################
Problem Statement: A logistics company recorded the time taken for delivery and the time taken for 
                   the sorting of the items for delivery. Build a Simple Linear Regression model to find 
                   the relationship between delivery time and sorting time with delivery time as the target variable. 
                   Apply necessary transformations and record the RMSE and 
                   correlation coefficient values for different models.
########################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as sfa
from sklearn.model_selection import train_test_split

# Reading a dataset into python
data = pd.read_csv(r"D:/Hands on/23_Simple Linear Regression/Assignment/delivery_time.csv")

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Columns of the dataset
data.columns

# Renaming the dataset columns
data.rename({'Delivery Time' : 'Delivery_Time', 
             'Sorting Time' : 'Sorting_Time'}, axis = 1, inplace = True)

# Columns of the dataset
data.columns

# Bar plot
plt.bar(data.Delivery_Time, height = np.arange(1, 22))
plt.bar(data.Sorting_Time, height = np.arange(1, 22))

# Histogram
plt.hist(data.Delivery_Time)
plt.hist(data.Sorting_Time)

# Boxplot
plt.boxplot(data.Delivery_Time)
plt.boxplot(data.Sorting_Time)

# Correlation coefficient
data.corr()
np.corrcoef(data.Delivery_Time, data.Sorting_Time)

# Scatter diagram
plt.scatter(data.Sorting_Time, data.Delivery_Time)


# Building a model
model1 = sfa.ols('Delivery_Time ~ Sorting_Time', data = data).fit()

# Summary of the model
model1.summary()

# Prediction
pred1 = model1.predict(pd.DataFrame(data['Sorting_Time']))

# Scatter plot along with predicted line
plt.scatter(data.Sorting_Time, data.Delivery_Time)
plt.plot(data.Sorting_Time, pred1)
plt.legend(['Observed values', 'Predicted values'])
plt.show()

# RMSE
err1 = data.Delivery_Time - pred1
serr1 = err1 * err1
mserr1 = np.mean(serr1)
rmse1 = np.sqrt(mserr1)
rmse1

# Scatter plot
plt.scatter(np.log(data.Sorting_Time), data.Delivery_Time)

# Correlation coefficient
np.corrcoef(np.log(data.Sorting_Time), data.Delivery_Time)

# Model Building
model2 = sfa.ols('Delivery_Time ~ np.log(Sorting_Time)', data = data).fit()

# Summary of the dataset
model2.summary()

# Prediction
pred2 = model2.predict(data.Sorting_Time)

# Scatter plot along with predicted line
plt.scatter(np.log(data.Sorting_Time), data.Delivery_Time)
plt.plot(np.log(data.Sorting_Time), pred2, 'r')
plt.legend(['Observed Values', 'Predicted Values'])
plt.show()

# RMSE
err2 = data.Delivery_Time - pred2
serr2 = err2 * err2
merr2 = np.mean(serr2)
rmse2 = np.sqrt(merr2)
rmse2

# Scatter plot 
plt.scatter(data.Sorting_Time, np.log(data.Delivery_Time))

# Correlation coefficient
np.corrcoef(data.Sorting_Time, np.log(data.Delivery_Time))            

# Model builing
model3 = sfa.ols('np.log(Delivery_Time) ~ Sorting_Time', data = data).fit()

# Summary of the model
model2.summary()

# Prediction
pred3 = model3.predict(data.Sorting_Time)
pred3 = np.exp(pred3)

# Scatter plot along with predicted line
plt.scatter(data.Sorting_Time, np.log(data.Delivery_Time))
plt.plot(data.Sorting_Time, pred3, 'r')
plt.legend(['Observed values', 'Predicted values'])
plt.show()

# RMSE
err3 = data.Delivery_Time - pred3
serr3 = err3 * err3
merr3 = np.mean(serr3)
rmse3 = np.sqrt(merr3)
rmse3

# Scatter plot
plt.scatter(data.Sorting_Time, np.sqrt(data.Delivery_Time))

# Correlation coefficient
np.corrcoef(np.sqrt(data.Sorting_Time), data.Delivery_Time)

# Model builind
model4 = sfa.ols('Delivery_Time ~ np.sqrt(Sorting_Time)', data = data).fit()

# Summary of the model
model4.summary()

# Prediction
pred4 = model4.predict(pd.DataFrame(data.Sorting_Time))

# RMSE
err4 = data.Delivery_Time - pred4
serr4 = err4 * err4
merr4 = np.mean(serr4)
rmse4 = np.sqrt(merr4)
rmse4

# Creating a dataframe
model_rmse = pd.DataFrame({ 'model' : pd.Series(['SLR', 'Log', 'Exp', 'Sqrt']), 
        'RMSE' : pd.Series([rmse1, rmse2, rmse3, rmse4])})

model_rmse

# Spliting the data into train test split
train, test = train_test_split(data, test_size =  0.2, random_state = 0)

# Model building
final_model = sfa.ols('Delivery_Time ~ Sorting_Time', data = data).fit()

# Summary of the model
final_model.summary()

# Prediction on train data
train_pred = final_model.predict(pd.DataFrame(train))

# Train data RMSE
err = train.Delivery_Time - train_pred
serr = err * err
merr = np.mean(serr)
train_rmse = np.sqrt(merr)

# Prediction on train data
test_pred = final_model.predict(pd.DataFrame(test))

# Test data RMSE
err = test.Delivery_Time - test_pred
serr = err * err
merr = np.mean(serr)
test_rmse = np.sqrt(merr)

#######################################################################################################
Problem Statement: A certain organization wants an early estimate of their employee churn out rate. 
So the HR department gathered the data regarding the employee’s salary hike and the churn out rate 
in a financial year. The analytics team will have to perform an analysis and predict an estimate of 
employee churn based on the salary hike. Build a Simple Linear Regression model with churn out rate 
as the target variable. Apply necessary transformations and record the RMSE and correlation coefficient 
values for different models.

########################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as sfa
from sklearn.model_selection import train_test_split

# Reading a dataset into python
data = pd.read_csv(r"D:/Hands on/23_Simple Linear Regression/Assignment/emp_data.csv")

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Columns of the dataset
data.columns

# Barplot
plt.bar(data.Salary_hike, height = range(1, 11, 1))
plt.bar(data.Churn_out_rate, height = range(1, 11, 1))

# Histogram
plt.hist(data.Salary_hike)
plt.hist(data.Churn_out_rate)

# Boxplot
plt.boxplot(data.Salary_hike)
plt.boxplot((data.Churn_out_rate))

# Scatter plot
plt.scatter(data.Salary_hike, data.Churn_out_rate)

# Correlation coefficient
np.corrcoef(data.Salary_hike, data.Churn_out_rate)

# Model building
model1 = sfa.ols('Churn_out_rate ~ Salary_hike', data = data).fit()

# Summary of the model
model1.summary()

# Prediction
pred1 = model1.predict(pd.DataFrame(data['Salary_hike']))

# Scatter plot along with predicted line
plt.scatter(data.Salary_hike, data.Churn_out_rate)
plt.plot(data.Salary_hike, pred1, 'r')
plt.legend(['Observed values', 'Predicted values'])
plt.show()

# RMSE
err1 = data.Churn_out_rate - pred1
serr1 = err1 * err1
merr1 = np.mean(serr1)
rmse1 = np.sqrt(merr1)
rmse1

# Scatter plot
plt.scatter(data.Salary_hike, np.log(data.Churn_out_rate))

# Correlation coefficient
np.corrcoef(data.Salary_hike, np.log(data.Churn_out_rate))

# Model Builiding
model2 = sfa.ols('np.log(Churn_out_rate) ~ Salary_hike', data = data).fit()

# Summary of the model
model2.summary()

# Prediction
pred2 = model2.predict(pd.DataFrame(data['Salary_hike']))
pred2_y = np.exp(pred2)

# Scatter plot along with predicted line
plt.scatter(data.Salary_hike, np.log(data.Churn_out_rate))
plt.plot(data.Salary_hike, pred2, 'r')
plt.legend(['Observed values', 'Predicted values'])
plt.show()

# RMSE
err2 = data.Churn_out_rate - pred2_y
serr2 = err2 * err2
merr2 = np.mean(serr2)
rmse2 = np.sqrt(merr2)
rmse2

# Split the data into train and test split
train, test = train_test_split(data, test_size = 0.20, random_state = 0)

# Model building on training data
final_model = sfa.ols('np.log(Churn_out_rate) ~ Salary_hike', data = train).fit()

# Summary of the model
final_model.summary()

# Train data prediction
train_pred = final_model.predict(pd.DataFrame(train))
train_pred1 = np.exp(train_pred)

# Train data RMSE
train_err = train.Churn_out_rate - train_pred1
strain_err = train_err * train_err
mtrain_err = np.mean(strain_err)
rmsetrain_err = np.sqrt(mtrain_err)

# Test data prediction
test_pred = final_model.predict(pd.DataFrame(test))
test_pred1 = np.exp(test_pred)

# Test data RMSE
test_err = test.Churn_out_rate - test_pred1
stest_err = test_err * test_err
mtest_err = np.mean(stest_err)
rmsetest_err = np.sqrt(mtest_err)

#######################################################################################################
Problem Statement: The head of HR of a certain organization wants to automate their salary hike 
estimation. The organization consulted an analytics service provider and asked them to build a basic 
prediction model by providing them with a dataset that contains the data about the number of years of 
experience and the salary hike given accordingly. Build a Simple Linear Regression model with salary 
as the target variable. Apply necessary transformations and record the RMSE and correlation coefficient 
values for different models.

#######################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as sfa
from sklearn.model_selection import train_test_split

# Reading a dataset into python
data = pd.read_csv(r"D:/Hands on/23_Simple Linear Regression/Assignment/Salary_Data.csv")

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Columns of the dataset
data.columns

# Barplot
plt.bar(data.YearsExperience, height = np.arange(1, 31, 1))
plt.bar(data.Salary, height = np.arange(1, 31, 1))

# Histogram
plt.hist(data.YearsExperience)
plt.hist(data.Salary)

# Boxplot
plt.boxplot(data.YearsExperience)
plt.boxplot((data.Salary))

# Scatter plot
plt.scatter(data.YearsExperience, data.Salary)

# Correlation coefficient
np.corrcoef(data.YearsExperience, data.Salary)

# Model building
model1 = sfa.ols('Salary ~ YearsExperience', data = data).fit()

# Summary of the model
model1.summary()

# Prediction
pred1 = model1.predict(pd.DataFrame(data['YearsExperience']))

# Scatter plot along with predicted line
plt.scatter(data.YearsExperience, data.Salary)
plt.plot(data.YearsExperience, pred1, 'r')
plt.legend(['Observed values', 'Predicted values'])
plt.show()

# RMSE
err1 = data.Salary - pred1
serr1 = err1 * err1
merr1 = np.mean(serr1)
rmse1 = np.sqrt(merr1)
rmse1

# Split the data into train and test split
train, test = train_test_split(data, test_size = 0.20, random_state = 0)

# Model building on training data
final_model = sfa.ols('Salary ~ YearsExperience', data = train).fit()

# Summary of the model
final_model.summary()

# Train data prediction
train_pred = final_model.predict(pd.DataFrame(train))


# Train data RMSE
train_err = train.Salary - train_pred
strain_err = train_err * train_err
mtrain_err = np.mean(strain_err)
rmsetrain_err = np.sqrt(mtrain_err)
rmsetrain_err 

# Test data prediction
test_pred = final_model.predict(pd.DataFrame(test))

# Test data RMSE
test_err = test.Salary - test_pred
stest_err = test_err * test_err
mtest_err = np.mean(stest_err)
rmsetest_err = np.sqrt(mtest_err)
rmsetest_err 
