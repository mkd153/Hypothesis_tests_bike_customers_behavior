import numpy as np, seaborn as sns, pandas as pd, math, matplotlib.pyplot as plt, scipy.stats
from scipy.stats import chi2, chi2_contingency, ttest_ind, f_oneway
import statsmodels
from statsmodels.stats.weightstats import ztest

bikes = pd.read_csv('data.csv')

bikes.head() # first 5 rows

bikes.shape # shape of dataset

bikes.info()  # information about columns, their data types, count

# There are some numerical variables that can be converted to categorical columns by changing the numbers to categories such as in season, holiday, workingday, and weather.

bikes.season.replace([1,2,3,4,],['spring','summer','fall','winter'],inplace=True)

bikes.season.value_counts()

bikes.holiday.replace([0,1],['no_holiday','holiday'],inplace=True) # converting numerical to categorical variable

bikes.holiday.value_counts()

# There are only 311 holiday entries in the dataset, most entries are from non holiday days. 

bikes.workingday.replace([0,1],['not_working','working'],inplace=True) # converting numerical to categorical variablev

bikes.workingday.value_counts()

bikes.weather.replace([1,2,3,4],['clear','misty','light_rain/snow','heavy_rain/snow'],inplace=True) # converting numerical to categorical variable

bikes.weather.value_counts()

bikes.describe() # statistical summary of all numerical variables


### Missing values and outlier detection

bikes.isnull().sum()

sns.boxplot(bikes.temp) # looking at the spread

sns.boxplot(bikes.atemp)

sns.boxplot(bikes.humidity)

sns.boxplot(bikes.windspeed)

sns.boxplot(bikes['count'])

iqr = scipy.stats.iqr(bikes['count']) # inter quartile range
q3 = np.percentile(bikes['count'],75) # third quartile
bikes['count'][bikes['count'] > (q3 +iqr*1.5)] # outlier points

sns.countplot(data=bikes, x = 'season') # all seasons have almost equal number of days

sns.histplot(data=bikes, x='temp')

sns.histplot(data=bikes, x='humidity')

sns.histplot(data=bikes, x='windspeed')

sns.histplot(data=bikes, x='count') 

sns.boxplot(data=bikes, x='season', y='count')

sns.boxplot(data=bikes, x='weather', y='count')

sns.boxplot(data=bikes, y='count', x='workingday')

sns.countplot(data=bikes, x='weather', hue='season')

## 4. Hypothesis testing

### 2- Sample T-Test to check if Working Day has an effect on the number of electric cycles rented

# T-test Ho: the means for counts of users for working day and non working day entries are equal.

# Ha: they are unequal

a = bikes[bikes['workingday']=='not_working']['count']
b = bikes[bikes['workingday']=='working']['count']
print(np.array(a).var(),np.array(b).var()) # variances are different for the two groups
ttest_ind(a,b)

### ANOVA to check if No. of cycles rented is similar or different in different weather 
#### Weather

#Ho : mean number of cycles rented per hour in different weather is similar

#Ha : mean number of cycles rented are different for different weather

a= bikes[bikes['weather']=='clear']['count']
b = bikes[bikes['weather']=='misty']['count']
c = bikes[bikes['weather']=='light_rain/snow']['count']
d = bikes[bikes['weather']=='heavy_rain/snow']['count']
print(np.array(a).var(),np.array(b).var(),np.array(c).var(),np.array(d).var()) # variances for the 4 groups
f_oneway(a,b,c,d)

### Chi-square test to check if Weather is dependent on the season

crosstable = pd.crosstab(index=bikes['weather'],columns=bikes['season'])
crosstable
#Ho: the proportion of different weather days is equal for all seasons.

#Ha: the proportions are not equal, weather days are dependent on seasons.

chi2_contingency(crosstable) 
