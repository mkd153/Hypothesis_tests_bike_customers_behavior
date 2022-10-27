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

# There are almost double working day entries compared to weekend and holiday entries.

bikes.weather.replace([1,2,3,4],['clear','misty','light_rain/snow','heavy_rain/snow'],inplace=True) # converting numerical to categorical variable

bikes.weather.value_counts()

# Let's look at the numerical variables

## 2. Univariate analysis

bikes.describe() # statistical summary of all numerical variables


### Missing values and outlier detection

bikes.isnull().sum()

# There are no missing values.

# Let us look at any outliers present in the numerical variables like temp, atemp, humidity, windspeed, casual, registered.

sns.boxplot(bikes.temp) # looking at the spread

# There are no outliers in temperature variable.

sns.boxplot(bikes.atemp)

sns.boxplot(bikes.humidity)

# There is an outlier at 0 humidity, which is a reasonable number for a dry day so we would not discard it.

sns.boxplot(bikes.windspeed)

# Windspeed seems to have a right skewed distribution which lots of large outliers, but they seem like high winds days and not errors so we would not discard them.

sns.boxplot(bikes['count'])

iqr = scipy.stats.iqr(bikes['count']) # inter quartile range
q3 = np.percentile(bikes['count'],75) # third quartile
bikes['count'][bikes['count'] > (q3 +iqr*1.5)] # outlier points

# There are 300 outliers all closely situated, and don't seem like error in measurement. So we would keep them.

sns.countplot(data=bikes, x = 'season') # all seasons have almost equal number of days

sns.histplot(data=bikes, x='temp')

# This looks quite normally distributed.

sns.histplot(data=bikes, x='humidity')

# Humidity looks slightly normal distribution but more higher humidity numbers are present.

sns.histplot(data=bikes, x='windspeed')

sns.histplot(data=bikes, x='count') # histogram of number of users per hour is almost linearly decreasing with increasing frequency, apart from 0 users at a time

# This looks slightly like an exponential distribution.

## 3. Bivariate analysis

sns.boxplot(data=bikes, x='season', y='count')

# Mean number of users who used a bike in an hour on a day is 191.



# The count distribution visually looks similar for summer, spring and highest for fall season for this sample, but less number of users per hour in spring than rest of the seasons in this sample dataset. For knowing how significantly different they are, we need to perform significance testing methods like ANOVA (analysis of variance).   

sns.boxplot(data=bikes, x='weather', y='count')

# The median counts of users per hour is highest for clear sky, slightly less for misty weather and quite less for light rain/snow weather days. There are hardly any users for heavy rain/snow days. We would perform ANOVA to check statistical significance.

sns.boxplot(data=bikes, y='count', x='workingday')

# The median number of users per hour on a working day and non working day are similar. To check statistical significance we need to perform 2-sample T-tests as the population standard deviation is unknown.

sns.countplot(data=bikes, x='weather', hue='season')

# The number of hours recorded in every season is similar for light rain/snow weather and heavy rain/snow weather. However, they are different for clear sky weather, misty weather. We can check the statistical significance using chi-square to check the proportions.

## 4. Hypothesis testing

### 2- Sample T-Test to check if Working Day has an effect on the number of electric cycles rented

# Some of the assumptions of the T-test are met such as random sampling, adequate number of samples. But some assumptions are not met such as Normality, Equal Variance.

# We would continue doing the analysis even if some assumptions fail.


# T-test Ho: the means for counts of users for working day and non working day entries are equal.

# Ha: they are unequal

a = bikes[bikes['workingday']=='not_working']['count']
b = bikes[bikes['workingday']=='working']['count']
print(np.array(a).var(),np.array(b).var()) # variances are different for the two groups
ttest_ind(a,b)

# T-test p-value 0.23 is larger than alpha=0.05. So we fail to reject the null hypothesis. Hence, similar to the visual analysis plots, working day has no statistically significant effect on the mean of number of cycles.

### ANOVA to check if No. of cycles rented is similar or different in different 1. weather 2. season

#### 1. Weather

#Ho : mean number of cycles rented per hour in different weather is similar

#Ha : mean number of cycles rented are different for different weather

a= bikes[bikes['weather']=='clear']['count']
b = bikes[bikes['weather']=='misty']['count']
c = bikes[bikes['weather']=='light_rain/snow']['count']
d = bikes[bikes['weather']=='heavy_rain/snow']['count']
print(np.array(a).var(),np.array(b).var(),np.array(c).var(),np.array(d).var()) # variances for the 4 groups
f_oneway(a,b,c,d)

#We assume that the variance for different population groups is same in order to perform F-test ANOVA, even though they differ largely.

#As the p-value (5.5e-42) is very low, we reject the null hypothesis with a 95% confidence level (alpha=0.05). We can say that the mean number of cycles differs for different weather days, they are not independent.

#### 2. Seasons

#Ho : mean number of cycles rented per hour in different seasons is similar

#Ha : mean number of cycles rented are different for different seasons

a= bikes[bikes['season']=='spring']['count']
b = bikes[bikes['season']=='summer']['count']
c = bikes[bikes['season']=='fall']['count']
d = bikes[bikes['season']=='winter']['count']
print(np.array(a).var(),np.array(b).var(),np.array(c).var(),np.array(d).var()) # variances for the 4 groups
f_oneway(a,b,c,d)

#We assume that the variances being different doesn't affect the ANOVA F-test.

#As the p-value is very low (6.2e-149), we reject the null hypothesis at 95% confidence level (alpha=0.05). We can say that the mean number of cycles differs for different seasons, they are not independent of each other.

### Chi-square test to check if Weather is dependent on the season



crosstable = pd.crosstab(index=bikes['weather'],columns=bikes['season'])
crosstable

#There are counts/frequencies for different weather in different seasons. They can be assumed to be independent of each other. So, we create a null hypothesis for chi-square test.

#Ho: the proportion of different weather days is equal for all seasons.

#Ha: the proportions are not equal, weather days are dependent on seasons.

chi2_contingency(crosstable) 


