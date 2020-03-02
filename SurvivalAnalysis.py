#dataset link
#https://www.kaggle.com/blastchar/telco-customer-churn/download

from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
import pandas as pd

## Read Data from csv
fileName = 'Telco-Customer-Churn.csv'
input_df = pd.read_csv(fileName)

## Replace yes and No in the Churn column to 1 and 0. 1 for the event and 0 for the censured data.
input_df['Churn']=input_df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0 )

## Convert TotalCharges to numeric
# input_df['TotalCharges']=pd.to_numeric(input_df['TotalCharges'],errors='coerce')

T = input_df['tenure']
E = input_df['Churn']
# print(T)

kmf = KaplanMeierFitter()
## Two Cohorts are compared.
# 1. Streaming TV Not Subscribed by users, and Cohort
# 2. Streaming TV subscribed by the users.
groups = input_df['StreamingTV']
i1 = (groups == 'No')      ## group i1 , having the pandas series  for the 1st cohort
i2 = (groups == 'Yes')     ## group i2 , having the pandas series  for the 2nd cohort

## fit the model for 1st cohort
kmf.fit(T[i1],E[i1], label='Not Subscribed StreamingTV')
a1 = kmf.plot()

## fit the model for 2nd cohort
kmf.fit(T[i2],E[i2], label='Subscribed StreamingTV')
kmf.plot(ax=a1)

## Two Cohorts are compared.
# 1. users not having partners in the same telco,
# 2. users having partners in the same telco
groups = input_df['Partner']
i1 = (groups == 'No')      ## group i1 , having the pandas series  for the 1st cohort
i2 = (groups == 'Yes')     ## group i2 , having the pandas series  for the 2nd cohort

## fit the model for 1st cohort
kmf.fit(T[i1],E[i1], label='No Partner')
a1 = kmf.plot()

## fit the model for 2nd cohort
kmf.fit(T[i2],E[i2], label='Partner')
kmf.plot(ax=a1)


#### 3 new cohorts are compared
# 1. Contract type is month-to-month
# 2. Contract type is Two
# 2. Contract type is One year
groups = input_df['Contract']             ## Create the cohorts from the 'Contract' column
ix1 = (groups == 'Month-to-month')   ## Cohort 1
ix2 = (groups == 'Two year')         ## Cohort 2
ix3 = (groups == 'One year')         ## Cohort 3

kmf.fit(T[ix1], E[ix1], label='Month-to-month')    ## fit the cohort 1 data
ax = kmf.plot()


kmf.fit(T[ix2], E[ix2], label='Two year')         ## fit the cohort 2 data
ax1 = kmf.plot(ax=ax)


kmf.fit(T[ix3], E[ix3], label='One year')        ## fit the cohort 3 data
kmf.plot(ax=ax1)                                 ## Plot the KM curve for three cohort on same x and y axis

print(kmf.predict(T[0]))
plt.show()

