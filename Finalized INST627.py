#!/usr/bin/env python
# coding: utf-8

# In[19]:


###Load Data
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/liushuyuan/Desktop/Dario/iSchool/INST627/RawData.csv')
df_population = pd.read_excel('/Users/liushuyuan/Desktop/Dario/iSchool/INST627/Maryland_DemographicsByCounty_sample.xlsx')
edu_data = pd.read_csv('/Users/liushuyuan/Desktop/Dario/iSchool/INST627/Enrollments by Gender and Races.csv')
earnings_data = pd.read_excel('/Users/liushuyuan/Desktop/Dario/iSchool/INST627/Earnings Disparity Race and Ethnicity Data.xlsx')
race_pop = pd.read_csv('/Users/liushuyuan/Desktop/Dario/iSchool/INST627/Race and Ethnicity.csv')
#earnings_data = pd.read_excel('/Users/liushuyuan/Desktop/Dario/iSchool/INST627/Earnings Disparity Race and Ethnicity Data.xlsx'
                              


# In[13]:


#Clean data by dropping NAN
#Check the assumption:
#1: Normality:
df_State_Values = df[df['Jurisdiction']== 'State']['Value']
#check for NaN values in the "State"
state_values_nan_check = df_State_Values.isna().sum()
state_values_nan_check
State_clean = df_State_Values.dropna()

df


# In[16]:


nan_check = df['Value'].isna().sum()
df_cleaned = df.dropna(subset = ['Value'])


# In[18]:


def merge_race_categories_er(race):
    if race == 'Black Non-Hispanic':
        return 'Black'
    elif race == 'White Non-Hispanic':
        return 'White'
    elif race == 'Asian/ Pacific Islander Non-Hispanic':
        return 'Asian/ Pacific Islander'
    else:
        return race
    
df_cleaned['Race/ ethnicity'] = df_cleaned['Race/ ethnicity'].apply(merge_race_categories_er)
df_cleaned_statelevel = df_cleaned.groupby(['Race/ ethnicity','Year','Jurisdiction'])['Value'].sum().reset_index()
df_cleaned_statelevel.head(20)

df_cleaned_ready = df_cleaned_statelevel.query("`Race/ ethnicity` != 'All races/ ethnicities (aggregated)'")
df_cleaned_ready


# In[20]:


race_pop.groupby(['Race', 'Year'])['Population'].sum().reset_index()

def merge_race_categories_pop(race):
    if race == 'Black or African American Alone':
        return 'Black'
    elif race == 'Hispanic or Latino':
        return 'Hispanic'
    elif race == 'Asian Alone':
        return 'Asian/ Pacific Islander'
    elif race == 'Native Hawaiian & Other Pacific Islander Alone':
        return 'Asian/ Pacific Islander'
    elif race == 'White Alone':
        return 'White'
    else:
        return race
earnings_data_ready = earnings_data.rename(columns={'Data Type': 'Race/ ethnicity'})    
race_pop_ready = race_pop.rename(columns = {'Race': 'Race/ ethnicity'})
race_pop_ready['Race/ ethnicity'] = race_pop_ready['Race/ ethnicity'].apply(merge_race_categories_pop)

race_pop_ready.head(50)
def update_race(row):
    if row['ID Ethnicity'] == 1:
        return 'Hispanic'
    else:
        return row['Race/ ethnicity']

race_pop_ready['Race/ ethnicity'] = race_pop_ready.apply(update_race, axis=1)

race_pop_ready.head(50)

allowed_races = ["White", "Black", "Hispanic", "Asian/ Pacific Islander"]
race_pop_filtered = race_pop_ready.query("`Race/ ethnicity` in @allowed_races")
race_pop_filtered.head(50)

Final_race_pop=race_pop_filtered.groupby(['Race/ ethnicity','Year'])['Population'].sum().reset_index()
Final_race_pop.head(50)


# In[21]:


earnings_data_ready = earnings_data.rename(columns={'Data Type': 'Race/ ethnicity'})
def merge_race_categories_earn(race):
    if race == 'Black Non-Hispanic':
        return 'Black'
    elif race == 'Hispanic/Latino':
        return 'Hispanic'
    elif race == 'Asian-Pacific Islander':
        return 'Asian/ Pacific Islander'
    else:
        return race
    
df_cleaned['Race/ ethnicity'] = df_cleaned['Race/ ethnicity'].apply(merge_race_categories_er)

md_earnings_data = earnings_data_ready[earnings_data_ready['State']=="MD"]
md_earnings_data['Race/ ethnicity']=md_earnings_data['Race/ ethnicity'].apply(merge_race_categories_earn)
# Remove unwanted race categories
md_earnings_data = md_earnings_data[
    ~md_earnings_data['Race/ ethnicity'].isin(['Multiracial', 'Native American/American Indian'])
]




# In[22]:


##education enrollment


# In[24]:


def merge_race_categories_edu(race):
    race_lower = race.lower()
    if 'hispanic' in race_lower or 'latino' in race_lower:
        return 'Hispanic'
    elif 'asian' in race_lower or 'pacific islander' in race_lower:
        return 'Asian/ Pacific Islander'
    elif 'black' in race_lower or 'african american' in race_lower:
        return 'Black'
    else:
        return race

edu_data['IPEDS Race'] = edu_data['IPEDS Race'].apply(merge_race_categories_edu)

# Now you can group by 'IPEDS Race' and 'ID Year' and sum the 'Enrollment' for each group
edu_data_new = edu_data.groupby(['IPEDS Race', 'Year'])['Enrollment'].sum().reset_index()

# Display the head of the new grouped DataFrame
edu_data_new.head(50)

# Remove unwanted race categories
edu_datacleaned_ready = edu_data_new[
    ~edu_data_new['IPEDS Race'].isin(['American Indian or Alaska Native', 'Non-resident Alien', 'Two or More Races', 'Unknown'])
]

edu_datacleaned_ready = edu_datacleaned_ready.rename(columns={'IPEDS Race': 'Race/ ethnicity'})


# In[30]:


#merge Data


merged_ER_edu = pd.merge(edu_datacleaned_ready, df_cleaned_ready, 
                     how='inner', 
                     on=['Year', 'Race/ ethnicity'])
merged_ER_edu.head(60)

merged_whole_data = pd.merge(merged_ER_edu,md_earnings_data, on='Race/ ethnicity')
merged_whole_data = merged_whole_data.rename(columns={'Value': 'ED visits', 'Enrollment': "University Enrollment"})
selected_columns = ["Race/ ethnicity", "Year","University Enrollment", "ED visits", "Average Weekly Earnings", "Employed Percent"]
ready_data = merged_whole_data[selected_columns]
ready_data.head(50)


# In[33]:


average_values_race =ready_data.groupby(['Race/ ethnicity','Year']).agg({
    'University Enrollment': 'mean',
    'ED visits': 'mean',
    'Average Weekly Earnings': 'mean',
    'Employed Percent': 'mean'
}).reset_index()
average_values_race.head(50)


# In[35]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# Convert 'Race/ ethnicity' to a categorical variable
average_values_race['Race/ ethnicity'].value_counts()
race_dummies = pd.get_dummies(average_values_race['Race/ ethnicity'], drop_first=True)


# In[40]:


average_values_race['intercept'] = 1
# Creating dummy variables
race_dummies = pd.get_dummies(average_values_race['Race/ ethnicity'], prefix='Race', drop_first=True)
#jurisdiction_dummies = pd.get_dummies(average_values_race['Jurisdiction'], prefix='Jurisdiction', drop_first=True)

# Combining the dummies with the original dataframe
data_with_dummies = pd.concat([average_values_race, race_dummies], axis=1)

# Define the model variables
explanatory_variables = ['intercept'] + list(race_dummies.columns)  + ['University Enrollment', 'Average Weekly Earnings', 'Employed Percent']
response_variable = 'ED visits'

# Handling NaN values by dropping or imputing
data_with_dummies = data_with_dummies.dropna(subset=explanatory_variables + [response_variable])

# Ensure that explanatory variables are numeric
for var in explanatory_variables:
    data_with_dummies[var] = pd.to_numeric(data_with_dummies[var], errors='coerce')

# Dropping rows with any NaNs in the model variables after conversion
data_with_dummies = data_with_dummies.dropna(subset=explanatory_variables + [response_variable])

# Fit the model
X = data_with_dummies[explanatory_variables]
y = data_with_dummies[response_variable]

model = sm.OLS(y, X).fit()
print(model.summary())


# In[45]:


#histogram
year_value = df_cleaned.groupby('Year')['Value'].sum().reset_index()



plt.figure(figsize=(12, 6))
plt.plot(year_value['Year'], year_value['Value'], marker = 'o')
plt.title("Total trend by year")
plt.xlabel('Year')
plt.ylabel('Total Value')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[46]:


# Convert 'Year' to a numeric type if it's not already.
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Create the pivot table.
pivot_data = df.pivot_table(index='Year', columns='Race/ ethnicity', values='Value', aggfunc='sum')

# Interpolate missing data points linearly.
pivot_data_interpolated = pivot_data.interpolate(method='linear', limit_direction='forward', axis=0)

# Now let's plot the data.
plt.figure(figsize=(12, 6))

for race in pivot_data_interpolated.columns:
    plt.plot(pivot_data_interpolated.index, pivot_data_interpolated[race], marker='o', label=race)

plt.title('Trend of Values by Race/Ethnicity Over Years')
plt.xlabel('Year')
plt.ylabel('Total Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[44]:


import matplotlib.pyplot as plt

race_ethnicity_grouped = df_cleaned.groupby('Race/ ethnicity')['Value'].sum()

# Convert the dictionary to a pandas Series, excluding 'All races/ethnicities (aggregated)'
race_ethnicity_series = pd.Series({key: val for key, val in race_ethnicity_grouped.items() if key != 'All races/ ethnicities (aggregated)'})

# Calculate percentages
race_ethnicity_percentage = (race_ethnicity_series / race_ethnicity_series.sum()) * 100

# Plotting the bar chart without 'All races/ethnicities (aggregated)'
plt.figure(figsize=(10, 8))
plt.bar(race_ethnicity_percentage.index, race_ethnicity_percentage.values, color='skyblue')
plt.title('Proportion of Each Race/Ethnicity (Excluding Aggregated)')
plt.ylabel('Percentage')
plt.xlabel('Race/Ethnicity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Output the percentages
race_ethnicity_percentage

print(race_ethnicity_series)


# In[52]:


# Filter out the aggregated category
filtered_data = df_cleaned[df_cleaned['Race/ ethnicity'] != 'All races/ ethnicities (aggregated)']

# Group the data by 'Race/ ethnicity' and get the list of values
grouped_values = [group['Value'].values for name, group in filtered_data.groupby('Race/ ethnicity')]

# Perform the ANOVA
anova_result = stats.f_oneway(*grouped_values)
print(anova_result)
                   


# In[48]:


#Boxplot for each jurisdication ED visit
filtered_values_dict = {}

# Loop through each unique jurisdiction in the DataFrame
for jurisdiction in df['Jurisdiction'].unique():
    # Extract the values for the current jurisdiction, removing any NaN values
    jurisdiction_values = df[df['Jurisdiction'] == jurisdiction]['Value'].dropna()
    
    # Calculate Q1, Q3, and IQR
    Q1 = jurisdiction_values.quantile(0.25)
    Q3 = jurisdiction_values.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Exclude outliers
    non_outliers = jurisdiction_values[(jurisdiction_values >= lower_bound) & (jurisdiction_values <= upper_bound)]
    filtered_values_dict[jurisdiction] = non_outliers

# Initialize the cleaned DataFrame outside the loop
cleaned_data_no_outliers = pd.DataFrame(columns=df.columns)

# Loop through the filtered values to append them to the cleaned DataFrame
for jurisdiction, values in filtered_values_dict.items():
    # Create a new DataFrame for the current jurisdiction without outliers
    temp_df = df[(df['Jurisdiction'] == jurisdiction) & (df['Value'].isin(values))]
    # Append the non-outlier rows to the cleaned DataFrame
    cleaned_data_no_outliers = cleaned_data_no_outliers.append(temp_df, ignore_index=True)

cleaned_data_no_outliers
value_lists = [values for values in filtered_values_dict.values()] 
labels = [jurisdiction for jurisdiction in filtered_values_dict.keys()]

plt.figure(figsize = (12,8))

plt.boxplot(value_lists, vert=False, labels = labels)
plt.title('Combined Boxplot for All Jurisdictions')
plt.xlabel('Values')
plt.tight_layout()  # Adjust the layout to fit all jurisdiction labels
plt.show()


# In[51]:


import scipy.stats as stats
# Create a list to hold the values for each jurisdiction
jurisdiction_values = [df[df['Jurisdiction'] == jurisdiction]['Value'].dropna()for jurisdiction in df["Jurisdiction"].unique()]
anova_result = stats.f_oneway(*jurisdiction_values)

print(anova_result)


# In[49]:


# the race composit for the lowest ED visits county
# Filter the data to exclude 'All races/ ethnicities (aggregated)'
prince_george_data_1 = df_cleaned[(df_cleaned['Jurisdiction'] == "Prince George's") & 
                                (df_cleaned['Race/ ethnicity'] != 'All races/ ethnicities (aggregated)')].dropna(subset=['Value'])

# Summing values for each race/ethnicity
race_sum = prince_george_data_1.groupby('Race/ ethnicity')['Value'].sum()
total_sum = race_sum.sum()

# Calculate percentages
race_percentage_1 = (race_sum / total_sum) * 100

# Plotting the bar chart
plt.figure(figsize=(12, 6))
plt.bar(race_percentage_1.index, race_percentage_1.values, color='skyblue')
plt.title("Percentage of Each Race/Ethnicity in Prince George's County (Excluding Aggregated)")
plt.ylabel('Percentage')
plt.xlabel('Race/Ethnicity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[58]:


# for last 10 year, find the ER visits in the Prince George, what is the racial composite?
prince_george_data_1 = df_cleaned[(df_cleaned['Jurisdiction'] == "Prince George's") & 
                                (df_cleaned['Race/ ethnicity'] != 'All races/ ethnicities (aggregated)')].dropna(subset=['Value'])

# Summing values for each race/ethnicity
race_sum = prince_george_data_1.groupby('Race/ ethnicity')['Value'].sum()
total_sum = race_sum.sum()

# Calculate percentages
race_percentage_1 = (race_sum / total_sum) * 100

# Plotting the bar chart
plt.figure(figsize=(12, 6))
plt.bar(race_percentage_1.index, race_percentage_1.values, color='skyblue')
plt.title("Percentage of Each Race/Ethnicity in Prince George's County (Excluding Aggregated)")
plt.ylabel('Percentage')
plt.xlabel('Race/Ethnicity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[60]:


# Assuming that 'Value' column contains individual observations
# Perform ANOVA
anova_result_no_Agg = stats.f_oneway(
    #prince_george_data[prince_george_data['Race/ ethnicity'] == 'All races/ ethnicities (aggregated)']['Value'],
    prince_george_data_1[prince_george_data_1['Race/ ethnicity'] == 'Asian/ Pacific Islander']['Value'],
    prince_george_data_1[prince_george_data_1['Race/ ethnicity'] == 'Black']['Value'],
    #prince_george_data_1[prince_george_data_1['Race/ ethnicity'] == 'Black Non-Hispanic']['Value'],
    prince_george_data_1[prince_george_data_1['Race/ ethnicity'] == 'Hispanic']['Value'],
    prince_george_data_1[prince_george_data_1['Race/ ethnicity'] == 'White']['Value'],
    #prince_george_data_1[prince_george_data_1['Race/ ethnicity'] == 'White Non-Hispanic']['Value']
    
)


anova_result_no_Agg


# In[63]:


import statsmodels.stats.multicomp as multi
PG_no_Agg = prince_george_data_1[prince_george_data_1['Race/ ethnicity'] != 'All races/ ethnicities (aggregated)']
tukey_PG = multi.pairwise_tukeyhsd(endog=PG_no_Agg['Value'],
                                  groups=PG_no_Agg['Race/ ethnicity'],
                                  alpha = 0.05)

tukey_PG.summary()


tukey_results_df = pd.DataFrame(data=tukey_PG._results_table.data[1:], columns=tukey_PG._results_table.data[0])

# Filter the results where the null hypothesis is rejected
PG_Rejected = tukey_results_df[tukey_results_df['reject'] == True]
PG_Rejected


# In[64]:


#F-Test: difference earning
f_value, p_value = stats.f_oneway(
    ready_data[ready_data['Race/ ethnicity'] == 'Asian/ Pacific Islander']['University Enrollment'],
    ready_data[ready_data['Race/ ethnicity'] == 'Black']['University Enrollment'],
    ready_data[ready_data['Race/ ethnicity'] == 'Hispanic']['University Enrollment'],
    ready_data[ready_data['Race/ ethnicity'] == 'White']['University Enrollment']
)

# Output the F-statistic and p-value
print('F-statistic:', f_value)
print('p-value:', p_value)


# In[65]:


tukey_f_edu = multi.pairwise_tukeyhsd(endog=ready_data['University Enrollment'],
                                  groups=ready_data['Race/ ethnicity'],
                                  alpha = 0.05)

summery_edu=tukey_f_edu.summary()


tukey_results_df = pd.DataFrame(data=tukey_f_edu._results_table.data[1:], columns=tukey_f_edu._results_table.data[0])

# Filter the results where the null hypothesis is rejected
Edu_Rejected = tukey_results_df[tukey_results_df['reject'] == True]

Edu_Rejected.head(26)


# In[67]:


#final chart
final_merged_data = pd.merge(ready_data, Final_race_pop, on=['Year', 'Race/ ethnicity'], how='left')

# Check the merged dataframe
# Drop rows where 'Population' is NaN
final_merged_data_cleaned = final_merged_data.dropna(subset=['Population'])

final_merged_data_cleaned.head(50)
#normality check:

from scipy.stats import pearsonr

# Assuming ready_data is your DataFrame
# Calculate the correlation coefficient between 'Population' and 'ED visits'
correlation_ed_visits, _ = pearsonr(final_merged_data_cleaned['Population'], final_merged_data_cleaned['ED visits'])

# Calculate the correlation coefficient between 'Population' and 'University Enrollment'
correlation_university_enrollment, _ = pearsonr(final_merged_data_cleaned['Population'], final_merged_data_cleaned['University Enrollment'])

print(f"Correlation between Population and ED visits: {correlation_ed_visits}")
print(f"Correlation between Population and University Enrollment: {correlation_university_enrollment}")


# In[ ]:




