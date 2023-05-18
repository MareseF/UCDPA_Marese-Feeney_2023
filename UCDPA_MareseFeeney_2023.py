#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# ## Import Libaries

# In[1]:


import numpy as np
import pandas as pd

# Regular Edxpression libary
import re

# machine learning libaries: Hypertuning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor ,plot_tree
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# machine learning libaries: Boosting
import xgboost as xgb

# visualisation libaries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as py


# ## Import Data

# In[2]:


#Importing 1st CSV file - Glassdoor Gender Pay Gap_1.csv
pay_gap1=pd.read_csv("Glassdoor Gender Pay Gap_1.csv")


# In[3]:


pay_gap1.head()


# In[4]:


#check the type of dataframe and if any null values in the datframe
pay_gap1.info()


# In[5]:


#check for regular null values
pay_gap1.isnull().any()


# In[6]:


pay_gap1.info()


# In[7]:


# no null values noted, if there had been, the following would have been used: 
pay_gap1= pay_gap1.fillna("unknown")


# In[8]:


#Importing 2nd CSV file - Glassdoor Gender Pay Gap_2.csv
pay_gap2=pd.read_csv("Glassdoor Gender Pay Gap_2.csv")


# In[9]:


# check the headings to ensure no disparity with first dataset
pay_gap2.head()


# In[10]:


pay_gap2.info()


# In[11]:


#check for regular null values
pay_gap2.isnull().any()


# # Analysing the Data

# ## Merging the dataframes

# In[12]:


#merge the dataframes before doing any further review or data clean-up 
pay_gap =  pd.concat(
    map(pd.read_csv, ['Glassdoor Gender Pay Gap_1.csv', 'Glassdoor Gender Pay Gap_2.csv']), ignore_index=True)


# In[13]:


#data observed from the two tables indicates that obnce combined, there should be 1,004 rows of data
pay_gap.info()


# ## Dropping Duplicates

# In[14]:


#Checking duplicates
pay_gap.duplicated().sum()


# In[15]:


#drop the duplicates
pay_gap = pay_gap.drop_duplicates().reset_index(drop=True)


# In[16]:


pay_gap.info()


# ## Cleaning the data

# In[17]:


#Combine the Basepay & Bonus pay to assist with analysis
pay_gap['TotalPay'] = pay_gap['BasePay'] + pay_gap['Bonus']


# In[18]:


pay_gap


# In[19]:


#rename index Column
pay_gap.index.rename('Ref_Number', inplace=True)


# In[20]:


pay_gap.shape


# In[21]:


pay_gap.info()


# In[22]:


pay_gap.head()


# In[23]:


#Review number of Males Vs Females levels and number of these
pay_gap.Gender.value_counts()


# In[24]:


#Review unique education levels and number of these
print('Education Level: ')
pay_gap.Education.value_counts()


# In[25]:


# count number of each education level by gender
education_counts = pay_gap.groupby('Gender')['Education'].value_counts()


# In[26]:


# print education levels by gender
print('Education Level by Gender:')
print(education_counts)


# In[27]:


# count number of each Performance evaluation by gender
perfeval_counts = pay_gap.groupby('Gender')['PerfEval'].value_counts()


# In[28]:


# print performance levels by gender
print('Performance Evaluation by Gender:')
print(perfeval_counts)


# In[29]:


# count number of each age level by gender
age_counts = pay_gap.groupby('Gender')['Age'].value_counts()


# In[30]:


# print age by gender
print('Age by Gender:')
age_counts


# In[31]:


pd.set_option('display.max_rows', None) 


# In[32]:


age_counts


# # Using Regex to extract a pattern in data

# In[33]:


# define function to get unique words from a column
def get_unique_words(pay_gap, JobTitle):
    all_text = ' '.join(pay_gap[JobTitle].astype(str).tolist())
    all_words = all_text.split()
    return set(all_words)


# In[34]:


# call function to identify unique words in jobtitles 
unique_words = get_unique_words(pay_gap, 'JobTitle')
print(f'The unique words in the "JobTitle" column are: {unique_words}')


# In[35]:


# define list of 15 unique words to search for
unique_words = ['Warehouse', 'Analyst', 'Financial', 'Associate', 'Marketing', 'Scientist', 'Designer', 'Sales', 'Manager', 'IT', 'Driver', 'Graphic', 'Data', 'Engineer', 'Software']


# In[36]:


# create function to count occurrences of each unique word in column
def count_words(pay_gap, JobTitle, words):
    counts = {word: 0 for word in words}
    for value in pay_gap[JobTitle]:
        for word in words:
            pattern = fr'\b{word}\b'
            matches = re.findall(pattern, value, flags=re.IGNORECASE)
            counts[word] += len(matches)
    return counts


# In[37]:


# call function and print results
word_counts = count_words(pay_gap, 'JobTitle', unique_words)
for word, count in word_counts.items():
    print(f'The word "{word}" appears {count} times in the JobTitle column.')


# # Use of iterators

# In[38]:


# Create an iterator for grouping the data by gender, age, education, and seniority
groups = pay_gap.groupby(['Gender', 'Age', 'Education', 'Seniority'])


# In[39]:


# Iterate over the groups and calculate the average pay for each group
for group, data in groups:
    avg_pay = data['TotalPay'].mean()
    print(f"Gender: {group[0]}, Age: {group[1]}, Education: {group[2]}, Seniority: {group[3]}, Average Pay: {avg_pay}")


# In[40]:


# Create an iterator for grouping the data by gender, JobTitle, age, education, and pay
groups = pay_gap.groupby(['Gender', 'JobTitle', 'Age', 'Education', 'TotalPay'])


# In[41]:


# Iterate over the groups and calculate the average pay for each group
for group, data in groups:
    avg_pay = data['TotalPay'].mean()
    print(f"Gender: {group[0]}, JobTitle:{group[1]}, Age: {group[2]}, Education: {group[3]}, Average Pay: {avg_pay}")


# In[42]:


# Create an iterator for grouping the data by gender, jobtitle, age, education, and performance
groups = pay_gap.groupby(['Gender', 'JobTitle', 'Age', 'Education', 'PerfEval'])


# In[43]:


# Iterate over the groups and calculate the average pay for each group
for group, data in groups:
    avg_pay = data['TotalPay'].mean()
    print(f"Gender: {group[0]}, JobTitle:{group[1]}, Age: {group[2]}, Education: {group[3]}, PerfEval: {group[4]}, Average Pay: {avg_pay}")


# # Define a custom code to create reusable code 

# In[44]:


def calculate_average_pay_by_gender(pay_gap):
    """
    Calculates the average pay for each gender.

    Parameters:
    - pay_gap: A Pandas DataFrame containing the gender pay data.

    Returns:
    - A new Pandas DataFrame with columns for gender and average total pay.
    """


# In[45]:


# Group the data by gender
groups = pay_gap.groupby('Gender')


# In[46]:


# Calculate the average pay for each group
avg_pays = groups.agg({'TotalPay': 'mean'}).reset_index()


# In[47]:


avg_pays.columns = ['Gender', 'Average Total Pay']


# In[48]:


avg_pays


# In[49]:


# Calculate the average bonus for each group
avg_pays2 = groups.agg({'Bonus': 'mean'}).reset_index()


# In[50]:


avg_pays2.columns = ['Gender', 'Bonus']


# In[51]:


avg_pays2


# In[52]:


# Calculate the average basepay for each group
avg_pays3 = groups.agg({'BasePay': 'mean'}).reset_index()


# In[53]:


avg_pays3.columns = ['Gender', 'BasePay']


# In[54]:


avg_pays3


# # Numpy

# ## Review the Mean, Median and Standard Deviation of Pay

# In[55]:


# Extract the Total pay column as a NumPy array
pay = np.array(pay_gap['TotalPay'])


# In[56]:


# Calculate the mean, median, and standard deviation of the pay data using NumPy
mean_pay = np.mean(pay)
median_pay = np.median(pay)
std_pay = np.std(pay)


# In[57]:


f"Mean pay: {mean_pay}"


# In[58]:


f"Median pay: {median_pay}"


# In[59]:


f"Standard deviation of pay: {std_pay}"


# ## Identify the number of Males and Females getting above or below the mean and median pay

# In[60]:


# Extract the TotalPay and Gender columns as NumPy arrays
total_pay = np.array(pay_gap['TotalPay'])
gender = np.array(pay_gap['Gender'])


# In[61]:


# Find the indices of males and females
male_indices = np.where(gender == 'Male')[0]
female_indices = np.where(gender == 'Female')[0]


# In[62]:


# Calculate the number of males and females getting above or below the mean pay
num_male_above_mean = np.sum(total_pay[male_indices] > mean_pay)
num_female_above_mean = np.sum(total_pay[female_indices] > mean_pay)
num_male_below_mean = np.sum(total_pay[male_indices] < mean_pay)
num_female_below_mean = np.sum(total_pay[female_indices] < mean_pay)


# In[63]:


'Number of Males Above Mean Pay:', num_male_above_mean


# In[64]:


'Number of Females Above Mean Pay:', num_female_above_mean


# In[65]:


'Number of Males Below Mean Pay:', num_male_below_mean


# In[66]:


'Number of Females Below Mean Pay:', num_female_below_mean


# In[67]:


# Given the variation in the data, calculate the number of males and females getting above or below the meadian pay
num_male_above_median = np.sum(total_pay[male_indices] > median_pay)
num_female_above_median = np.sum(total_pay[female_indices] > median_pay)
num_male_below_median = np.sum(total_pay[male_indices] < median_pay)
num_female_below_median = np.sum(total_pay[female_indices] < median_pay)


# In[68]:


'Number of Males Above Median Pay:', num_male_above_median


# In[69]:


'Number of Males Below Median Pay:', num_male_below_median


# In[70]:


'Number of Females Above Median Pay:', num_female_above_median


# In[71]:


'Number of Females Below Median Pay:', num_female_below_median


# # Use Of Dictionaries or Lists

# In[72]:


#List usage demonstrated as part of Regex function, so use of dictionaries detailed here


# In[73]:


# Convert the DataFrame to a list of dictionaries
data = pay_gap.to_dict('records')


# In[74]:


# Filter the data to get only the rows where gender is female
female_data = [row for row in data if row['Gender'] == 'Female']


# In[75]:


# Calculate the average pay for female employees
female_pay = [row['TotalPay'] for row in female_data]
avg_female_pay = sum(female_pay) / len(female_pay)


# In[76]:


f"Average pay for female employees: {avg_female_pay}"


# In[77]:


# Filter the data to get only the rows where gender is male
male_data = [row for row in data if row['Gender'] == 'Male']


# In[78]:


# Calculate the average pay for male employees
male_pay = [row['TotalPay'] for row in male_data]
avg_male_pay = sum(male_pay) / len(male_pay)


# In[79]:


f"Average pay for male employees: {avg_male_pay}"


# # Visualise

# In[80]:


for i in pay_gap:
  plt.title(i)
  sns.countplot(x=pay_gap[i])
  
  plt.xticks(rotation=90)
  plt.show()


# In[81]:


title = pd.get_dummies(pay_gap, columns=['Gender']).groupby('JobTitle').sum()

female = go.Pie(labels=title.index,values=title['Gender_Female'],name="Female",hole=0.5,domain={'x': [0,0.46]})
male = go.Pie(labels=title.index,values=title['Gender_Male'],name="Male",hole=0.5,domain={'x': [0.52,1]})

layout = dict(title = 'Job Title Distribution', font=dict(size=14), legend=dict(orientation="h"),
              annotations = [dict(x=0.2, y=0.5, text='Female', showarrow=False, font=dict(size=20)),
                             dict(x=0.8, y=0.5, text='Male', showarrow=False, font=dict(size=20)) ])

fig = dict(data=[female, male], layout=layout)
py.iplot(fig)


# In[82]:


education = pd.get_dummies(pay_gap, columns=['Gender']).groupby('Education').sum()

female = go.Pie(labels=education.index,values=education['Gender_Female'],name="Female",hole=0.5,domain={'x': [0,0.46]})
male = go.Pie(labels=education.index,values=education['Gender_Male'],name="Male",hole=0.5,domain={'x': [0.52,1]})

layout = dict(title = 'Education Distribution', font=dict(size=14), legend=dict(orientation="h"),
              annotations = [dict(x=0.2, y=0.5, text='Female', showarrow=False, font=dict(size=20)),
                             dict(x=0.8, y=0.5, text='Male', showarrow=False, font=dict(size=20)) ])

fig = dict(data=[female, male], layout=layout)
py.iplot(fig)


# In[83]:


seniority = pd.get_dummies(pay_gap, columns=['Gender']).groupby('Seniority').sum()

female = go.Pie(labels=seniority.index,values=seniority['Gender_Female'],name="Female",hole=0.5,domain={'x': [0,0.46]})
male = go.Pie(labels=seniority.index,values=seniority['Gender_Male'],name="Male",hole=0.5,domain={'x': [0.52,1]})

layout = dict(title = 'Seniority Level Distribution', font=dict(size=14), legend=dict(orientation="h"),
              annotations = [dict(x=0.2, y=0.5, text='Female', showarrow=False, font=dict(size=20)),
                             dict(x=0.8, y=0.5, text='Male', showarrow=False, font=dict(size=20)) ])

fig = dict(data=[female, male], layout=layout)
py.iplot(fig)


# In[84]:


# Set font scale to 1.5 to increase font size
sns.set(font_scale=1.5)

plt.figure(figsize=(20, 20))
sns.countplot(data=pay_gap, x="Gender", hue="JobTitle")

# Increase x and y axis label font size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Increase legend font size
plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=len(pay_gap['JobTitle'].unique()), fontsize=5)

plt.show()


# In[85]:


# need to adjust the chart so that it is easier to interpret
# Set font scale to 1.5 
sns.set(font_scale=1.5)

plt.figure(figsize=(20, 20))
sns.countplot(data=pay_gap, x="JobTitle", hue="Gender")

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, fontsize=16)

# Increase y axis label font size
plt.yticks(fontsize=16)

# Increase legend font size
plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=len(pay_gap['JobTitle'].unique()), fontsize=17)

plt.show()


# In[86]:


sns.countplot(data=pay_gap, x='Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Frequency of Gender')
plt.show()


# In[87]:


# Create two histograms, one for each gender
plt.figure(figsize=(10, 8))  
plt.hist(pay_gap[pay_gap['Gender'] == 'Male']['TotalPay'], alpha=0.5, label='Male')
plt.hist(pay_gap[pay_gap['Gender'] == 'Female']['TotalPay'], alpha=0.5, label='Female')
plt.xlabel('TotalPay')
plt.ylabel('Frequency')
plt.title('Distribution of TotalPay by Gender')
plt.legend()
plt.show()


# In[88]:


plt.figure(figsize=(10, 8))  
sns.scatterplot(data=pay_gap, x='Age', y='TotalPay', hue='Gender')
plt.xlabel('Age')
plt.ylabel('TotalPay')
plt.title('Age vs TotalPay')
plt.show()


# # Machine Learning

# ## Hyper Parameter Tuning

# ### Predicting Total Pay using Linear Regression

# In[89]:


# Select features and target variable
X = pay_gap[['Age', 'PerfEval', 'Seniority', 'BasePay', 'Bonus']]
y = pay_gap['TotalPay']


# In[90]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[91]:


# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[92]:


# Evaluate the model on the test set using R^2 score
score = model.score(X_test, y_test)
print("Model accuracy:", score)


# In[93]:


pay_gap.info()


# In[94]:


#identify the relevant colums for the training model
selected_columns = ['Gender', 'Age', 'Education','Dept', 'Seniority', 'BasePay', 'Bonus','TotalPay']


# In[95]:


cols_to_encode = selected_columns


# In[96]:


# use One hot encode to deal with the cateogorical colums 
encoder = pd.get_dummies(pay_gap[cols_to_encode])
encoder.head()


# In[97]:


encoder.info()


# In[98]:


#train linear regression model on the training set and evaluate its performance on the test set
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Model accuracy:", r2_score(y_test, y_pred))


# In[99]:


encoder.info()


# In[100]:


# change the Data type from unit8 to int64
encoder['Gender_Female'] = encoder['Gender_Female'].astype('int64')
encoder['Gender_Male'] = encoder['Gender_Male'].astype('int64')
encoder['Education_College'] = encoder['Education_College'].astype('int64')
encoder['Education_High School'] = encoder['Education_High School'].astype('int64')
encoder['Education_Masters'] = encoder['Education_Masters'].astype('int64')
encoder['Education_PhD'] = encoder['Education_PhD'].astype('int64')
encoder['Dept_Administration'] = encoder['Dept_Administration'].astype('int64')
encoder['Dept_Engineering'] = encoder['Dept_Engineering'].astype('int64')
encoder['Dept_Management'] = encoder['Dept_Management'].astype('int64')
encoder['Dept_Operations'] = encoder['Dept_Operations'].astype('int64')
encoder['Dept_Sales'] = encoder['Dept_Sales'].astype('int64')


# In[101]:


encoder.info()


# ### Predicting Gender pay using Random Forest

# In[102]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[103]:


# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[104]:


# Evaluate the model on the test set
score = model.score(X_test, y_test)
print("Model accuracy:", score)


# ## Boosting

# In[105]:


# Split the dataset into features (X) and target variable (y)
X = encoder.drop('TotalPay', axis=1)
y = encoder['TotalPay']


# In[106]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[107]:


# Create the XGBoost model
xgb_model = xgb.XGBRegressor()


# In[108]:


# Train the model
xgb_model.fit(X_train, y_train)


# In[109]:


# Predict on the test set
y_pred = xgb_model.predict(X_test)


# In[110]:


# Calculate root mean squared error (RMSE) as the evaluation metric
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)


# In[111]:


# scatter plot to visualize the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()


# In[113]:


#Calculate the mean absolute error (MAE)
mae = np.mean(np.abs(y_test - y_pred))
print("Mean Absolute Error (MAE):", mae)


# In[114]:


#Calculate the R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)


# In[ ]:




