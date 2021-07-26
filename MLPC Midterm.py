# import pandas
import pandas as pd

# import numpy for working with numbers
import numpy as np

# import plots
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# import data from Excel csv sheet
df = pd.read_csv(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\LifeCycleSavings V1.02.csv')

# show first 5 records of dataset
df.head()

# return the object type, which is dataframe
type(df)

# summarise main characteristics by displaying the summary statistics of the
    # attributes, including measures of central tendency, and measures of dispersion
df.describe()

# import searborn library for more variety of data visualisation using fewer 
    # syntax and interesting default themes
import seaborn as sns 

# compare linear relationships between attributes using correlation coefficient 
    # generated using correlation matrix
sns.heatmap(df.corr(), cmap = 'PuBu', annot = True)
plt.show()

# ---------------------------------- NOISES -----------------------------------

# identify missing values
# display the number of entries, the number of names of the column attributes,
    # the data type and digit placings, and the memory space used
df.info()

# identify extreme and impossible values, and outliers (anomalies), using boxplot
axis_font = {'size':'30'}
df.boxplot(rot = 0, boxprops = dict(color = 'blue'), return_type = 'axes', 
           figsize = (35, 5), vert = False)
plt.title("Box Plot of Intercountry Life-Cycle Savings Data", **axis_font) # title of plot
plt.suptitle("")
plt.xticks(**axis_font)
plt.yticks(**axis_font)
plt.ylabel("Attribute", **axis_font) # x axis label
plt.xlabel("Measurements (units)", **axis_font) # y axis label
plt.show()

# summary statistics of the attributes, including measures of central tendency
    # and measures of dispersion
df.describe()

# ---------------------------------- ERRORS -----------------------------------

# detect duplicated records
df[df.duplicated(subset = None, keep = False)]

# ------------------------------ DATA PREPROCESSING ---------------------------

# -------------------------------- DATA CLEANING ------------------------------

# fill in missing value with the mean value of the attribute
df['PopulationOver75'] = df['PopulationOver75'].fillna(df['PopulationOver75'].mean())
df['Income'] = df['Income'].fillna(df['Income'].mean())
df['GrowthRate'] = df['GrowthRate'].fillna(df['GrowthRate'].mean())
df.info()



# smooth outliers for 'PersonalSaving' using winsorization technique
# replace outlier with maximum or minimum non-outlier 

# compute interquartile range (IQR)
IQR = df['PersonalSaving'].quantile(0.75) - df['PersonalSaving'].quantile(0.25)

# compute maximum and minimum non-outlier value
minAllowed = df['PersonalSaving'].quantile(0.25)-1.5*IQR
maxAllowed = df['PersonalSaving'].quantile(0.75)+1.5*IQR

# replace outlier values
for i in range(len(df['PersonalSaving'])): 
    if df['PersonalSaving'][i] < minAllowed:
       df['PersonalSaving'] = df['PersonalSaving'].replace(df['PersonalSaving'][i], 
                                                           minAllowed)
    elif df['PersonalSaving'][i] > maxAllowed:
       df['PersonalSaving'] = df['PersonalSaving'].replace(df['PersonalSaving'][i], 
                                                           maxAllowed)
    else: continue



# smooth outliers for 'Income' using winsorization technique
# replace outlier with maximum or minimum non-outlier 

# compute interquartile range (IQR)
IQR = df['Income'].quantile(0.75) - df['Income'].quantile(0.25)

# compute maximum and minimum non-outlier value
minAllowed = df['Income'].quantile(0.25)-1.5*IQR
maxAllowed = df['Income'].quantile(0.75)+1.5*IQR

# replace outlier values
for i in range(len(df['Income'])): 
    if df['Income'][i] < minAllowed:
       df['Income'] = df['Income'].replace(df['Income'][i], minAllowed)
    elif df['Income'][i] > maxAllowed:
       df['Income'] = df['Income'].replace(df['Income'][i], maxAllowed)
    else: continue



# smooth impossible values of GrowthRate
df['GrowthRate'] = df['GrowthRate'].replace(219, 2.19)
df['GrowthRate'] = df['GrowthRate'].replace(299, 2.99)

# smooth outliers for 'GrowthRate' using winsorization technique
# replace outlier with maximum or minimum non-outlier 

# compute interquartile range (IQR)
IQR = df['GrowthRate'].quantile(0.75) - df['GrowthRate'].quantile(0.25)

# compute maximum and minimum non-outlier value
minAllowed = df['GrowthRate'].quantile(0.25)-1.5*IQR
maxAllowed = df['GrowthRate'].quantile(0.75)+1.5*IQR

# replace outlier values
for i in range(len(df['GrowthRate'])): 
    if df['GrowthRate'][i] < minAllowed:
       df['GrowthRate'] = df['GrowthRate'].replace(df['GrowthRate'][i], minAllowed)
    elif df['GrowthRate'][i] > maxAllowed:
       df['GrowthRate'] = df['GrowthRate'].replace(df['GrowthRate'][i], maxAllowed)
    else: continue



# confirm smoothed impossible values and outliers using boxplot
axis_font = {'size':'30'}
df.boxplot(rot = 0, boxprops = dict(color = 'blue'), return_type = 'axes', 
           figsize = (35, 5), vert = False)
plt.title("Box Plot of Intercountry Life-Cycle Savings Data", **axis_font) # title of plot
plt.suptitle("")
plt.xticks(**axis_font)
plt.yticks(**axis_font)
plt.ylabel("Attribute", **axis_font) # x axis label
plt.xlabel("Measurements (units)", **axis_font) # y axis label
plt.show()

# summary statistics of the attributes, including measures of central tendency
    # and measures of dispersion
df.describe()

# drop duplicated records, retain only one copy for each
df = pd.DataFrame.drop_duplicates(df)
df.shape
# 50 unique records for 5 attributes

# ----------------------------- DATA TRANSFORMATION ---------------------------

# make a duplicate copy of dataset before data transformation
df_ori = df

# normalise data using min-max scaling between 0 and 1

# define min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# transform data
df = scaler.fit_transform(df)

# convert array to dataframe, define attribute names
df = pd.DataFrame(df, columns=['PersonalSaving', 'PopulationUnder15',
                               'PopulationOver75', 'Income', 'GrowthRate'])

# confirm min max scaled
df.describe()

# display the number of entries, the number of names of the column attributes,
    # the data type and digit placings, and the memory space used
df.info()

# -------------------------------- DATA REDUCTION -----------------------------

# concept hierarchy generation and segmentation by natural partitioning of 
    # GrowthRate into three higher level concepts
for i in range(50): 
    # High level represented by level 3
    if df['GrowthRate'][i] > 0.6667:
        df['GrowthRate'] = df['GrowthRate'].replace(df['GrowthRate'][i], 3)
    # Middle level represented by level 2
    elif df['GrowthRate'][i] > 0.3334: 
        df['GrowthRate'] = df['GrowthRate'].replace(df['GrowthRate'][i], 2)
    # Low level represented by level 1
    elif df['GrowthRate'][i] < 0.3334:
        df['GrowthRate'] = df['GrowthRate'].replace(df['GrowthRate'][i], 1)
    else: continue

# set GrowthRate as the target class object
df['GrowthRate'] = df.GrowthRate.astype(int)
df['GrowthRate'] = df.GrowthRate.astype(str)
df['GrowthRate'] = df.GrowthRate.astype(object)

# display the number of entries, the number of names of the column attributes,
    # the data type and digit placings, and the memory space used
df.describe()

# compare linear relationships between attributes using correlation coefficient 
    # generated using correlation matrix
sns.heatmap(df.corr(), cmap = 'PuBu', annot = True)
plt.show()

# visualise pairs plot or scatterplot matrix in relation to GrowthRate level
g = sns.pairplot(df, hue = 'GrowthRate', palette = 'PuBu')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)

# display the number of entries, the number and names of the column attributes, 
    # the data type and digit placings, and the memory space used
df.info()

# list and count the target class label names and their frequency
from collections import Counter
count = Counter(df['GrowthRate'])
count.items()

# count of each target class label
plt.figure(figsize = (5, 5))
ax = sns.countplot(df['GrowthRate'], palette = 'PuBu')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, ha = "right")
plt.suptitle("Count of GrowthRate Levels")
plt.show()

# ---------------- CLASSIFICATION PREDICTIVE MODEL DEVELOPMENT ----------------

# classify and model the data using Decision Tree (DT) machine learning algorithm

# import train test split module
from sklearn.model_selection import train_test_split

# import DT algorithm from DT class
from sklearn.tree import DecisionTreeClassifier

# split dataset into attributes and labels
X = df.iloc[:, :-1].values # the attributes
y = df.iloc[:, 4].values # the labels

# choose appropriate range of training set proportions
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# decision tree based on entropy information gain, best splitter, and minimum 2 
    # sample leaves
DT = DecisionTreeClassifier(splitter = 'best', criterion = 'entropy', 
                            min_samples_leaf = 2)

# find best training set proportion for the chosen models
plt.figure()
for s in t:
    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s, 
                                                            random_state = 111)
        DT.fit(X_train, y_train) # consider DT scores
        scores.append(DT.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
plt.xlabel('Training Set Proportion') # x axis label
plt.ylabel('Accuracy'); # y axis label

# choose train test splits from original dataset as 70% train data and 
    # 30% test data for highest accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 111)

# number of records in training set
len(X_train)

# count each GrowthRate level in training set
count = Counter(y_train)
print(count.items())

# using Decision Tree (DT) classifier
classifierDT = DecisionTreeClassifier(splitter = 'best', criterion='entropy', 
                                      min_samples_leaf = 2)
classifierDT.fit(X_train, y_train)

# plot decison tree
from sklearn import tree
fig = plt.figure(figsize = (35, 30))
fn = ['PersonalSaving', 'PopulationUnder15', 'PopulationOver75', 'Income']
DT = tree.plot_tree(classifierDT, feature_names = fn, class_names = y, 
                    filled = True)

# outputs all extracted rules
rules = tree.export_text(classifierDT, feature_names = fn)
print(rules)

# identifies the important features
classifierDT.feature_importances_

# ----------------------- MODEL PERFORMANCE EVALUATION ------------------------

# number of records in test set
len(X_test)

# count each GrowthRate level in test set
count = Counter(y_test)
print(count.items())

# use the chosen three models to make predictions on test data
y_predDT = classifierDT.predict(X_test)

# using confusion matrix
# import classification report and confusion matrix function from their modules
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predDT))
print(classification_report(y_test, y_predDT))

# import accuracy score function from its module
from sklearn.metrics import accuracy_score

# using accuracy performance metric
print("Train Accuracy: ", accuracy_score(y_train, classifierDT.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predDT))

# data to plot
n_groups = 1
algorithms = ('Decision Tree (DT)')
train_accuracy = (accuracy_score(y_train, classifierDT.predict(X_train))*100)
test_accuracy = (accuracy_score(y_test, y_predDT)*100)

# create plot
fig, ax = plt.subplots(figsize=(2, 5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8
# train accuracy
rects1 = plt.bar(index, train_accuracy, bar_width, alpha = opacity, 
                 color='Cornflowerblue', label='Train')
# test accuracy
rects2 = plt.bar(index + bar_width, test_accuracy, bar_width, alpha = opacity, 
                 color='Teal', label='Test')
plt.xlabel('Decision Tree (DT) Algorithm') # x axis label
plt.ylabel('Accuracy (%)') # y axis label
plt.ylim(0, 100) # y axis limit
plt.title('Comparison of Algorithm Accuracies') # plot title
plt.xticks(index + bar_width * 0.5, algorithms) # x axis data labels
plt.legend(loc = 'upper right') # show legend

plt.show()

# ------------------------ ESTIMATES OF MODEL PREDICTIONS ---------------------

# new data record must be within the data ranges to avoid extrapolation
df_ori.describe()

# new data
newdata = [[12, 23, 4.5, 1500]]

# transform data
newdata = scaler.fit_transform(newdata)

# compute probabilities of assigning to each of the three classes of GrowthRate
probaDT = classifierDT.predict_proba(newdata)
probaDT.round(4) # round probabilities to four decimal places, if applicable

# make prediction of class label
predDT = classifierDT.predict(newdata)
predDT