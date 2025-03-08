# %% [markdown]
# # In 2020, how did higher percentages of frequent physical or mental distress moderate the relationship between environmental/ socioeconomic factors and COVID19 cases? 

# %%

#! pip install -q pandas
#! pip install -q pyppeteer
#! pip install -q qeds
#%pip install numpy
#! pip install -q pandas
#! pip install -q pyppeteer
#! pip install -q qeds
#! pip install -q matplotlib
#! pip install -q tabulate
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import qeds

# %% [markdown]
# ## Introduction
# 
# - Source of Data
#     * My dataset comes from a combination of data sources. The socioeconomic factors data comes from the US CDC metrics of "social vulnerability" and natural and man-made disasters. The weather data comes from the NOAA Global Surface Summary of the day, which links each county to a nearby weather station. Also, my dataset uses the County Health Rankings Data a dataset that contains many measures of community health. 
# - Background
#     * Many studies have studied the link between weather and COVID19 cases. The consensus is usually that warmer weather leads to more cases, but the issue is that socioeconomic status could be different in warmer areas and that could be a confounder. In my research, I want to affirm that relationship plus explore an additional aspect of consideration known as physical and mental stress. I want to see how this factor interacts with what we already know. 
# - Research Question
#     * In 2020, how did higher percentages of frequent physical or mental distress moderate the relationship between environmental/socioeconomic factors and COVID19 cases across US counties, specifically California?
# - Findings
#     * (put this part later)
# - Citations
#     * Benita, F., Rebollar-Ruelas, L., & Gaytán-Alfaro, E. D. (2022). What have we learned about socioeconomic inequalities in the spread of COVID-19? A systematic review. Sustainable cities and society, 86, 104158. https://doi.org/10.1016/j.scs.2022.104158
#     * Han, Y., Zhao, W., & Pereira, P. (2022). Global COVID-19 pandemic trends and their relationship with meteorological variables, air pollutants and socioeconomic aspects. Environmental research, 204(Pt C), 112249. https://doi.org/10.1016/j.envres.2021.112249
#     * Sarkodie, S. A., & Owusu, P. A. (2021). Global effect of city-to-city air pollution, health conditions, climatic & socio-economic factors on COVID-19 pandemic. The Science of the total environment, 778, 146394. https://doi.org/10.1016/j.scitotenv.2021.146394
#     * Ahmed, J., Jaman, M. H., Saha, G., & Ghosh, P. (2021). Effect of environmental and socio-economic factors on the spreading of COVID-19 at 70 cities/provinces. Heliyon, 7(5), e06979. https://doi.org/10.1016/j.heliyon.2021.e06979 
#     * Zhang, X., Maggioni, V., Houser, P., Xue, Y., & Mei, Y. (2022). The impact of weather condition and social activity on COVID-19 transmission in the United States. Journal of environmental management, 302(Pt B), 114085. https://doi.org/10.1016/j.jenvman.2021.114085
#     * Ganslmeier, M., Furceri, D., & Ostry, J. D. (2021). The impact of weather on COVID-19 pandemic. Scientific reports, 11(1), 22027. https://doi.org/10.1038/s41598-021-01189-3
# - Variables
#     * X-variables = mean_temp, dewpoint, windspeed, percent_frequent_physical_distress, percent_frequent_mental_distress, percent_no_highschool_diploma, average_daily_pm2_5, unemployment rate, high_school_graduation_rate, percent_limited_access_to_healthy_foods, personal income, real_gdp_2020
#     * Y-variable = Cases
# - Rationale
#     * I chose these variables because of their possible connection to COVID19 cases. Some studies like Ganslmeier et al. (2021) suggest a negative correlation between temperature, humidity, windspeed to COVID19 cases and I wish to explore that correlation with my temperature, dewpoint, and wind speed variables. Also, since Han et al. (2020) mention that the interaction of different types of pollution has a strong impact on cases, I included the PM2.5 variable to explore that effect. The physical and mental distress variables are there to add an additional vector for exploration. I wonder how those variables can interact with the already explored effects of other variables on COVID19 cases. Then I included some socioeconomic variables to cover their potential effect on COVID19 cases. For example, Benita et al. (2022) concludes that income inequality is positively correlated with transmission of the virus. I included education variables, measures of personal income, GDP, unemployment rate as socioeconomic variables. I think it is valuable to explore this since many research have stated there are limitations reducing the accuracy of their research.
# 
# 

# %% [markdown]
# ## Data Cleaning and Loading

# %%
# Load datasets and do some filtering 


# Load main data and filter for California only
covid_health_weather = pd.read_csv("c:/Users/Pikachu/OneDrive - University of Toronto/W 2024/ECO225/ECO225 Project/eco225-project-code-r-lee1888/Data/US_counties_COVID19_health_weather_data.csv")
covid_health_weather = covid_health_weather[covid_health_weather['state'] == 'California']

# Get the number of rows and columns in the dataframe
num_rows, num_columns = covid_health_weather.shape
print(f'The dataframe has {num_rows} rows and {num_columns} columns.')


# Load the data for California GDP and Income and filter for 2020 only
# Remove all year columns except for 2020
ca_gdp_2020 = pd.read_csv("c:/Users/Pikachu/OneDrive - University of Toronto/W 2024/ECO225/ECO225 Project/eco225-project-code-r-lee1888/Data/CAGDP1_CA_2001_2023.csv")
ca_gdp_2020 = ca_gdp_2020[['GeoFIPS', 'GeoName', 'Region', 'TableName', 'LineCode', 'IndustryClassification', 'Description', 'Unit', '2020']]

ca_inc_2020 = pd.read_csv("c:/Users/Pikachu/OneDrive - University of Toronto/W 2024/ECO225/ECO225 Project/eco225-project-code-r-lee1888/Data/CAINC1_CA_1969_2023.csv")
ca_inc_2020 = ca_inc_2020[['GeoFIPS', 'GeoName', 'Region', 'TableName', 'LineCode', 'IndustryClassification', 'Description', 'Unit', '2020']]

ca_unemp_2020 = pd.read_csv("c:/Users/Pikachu/OneDrive - University of Toronto/W 2024/ECO225/ECO225 Project/eco225-project-code-r-lee1888/Data/UnemploymentReport.csv")
ca_unemp_2020 = ca_unemp_2020[['FIPS', 'Name', '2020']]

ca_gdp_2020 = ca_gdp_2020[ca_gdp_2020['LineCode'] == 1]
ca_pop_2020 = ca_inc_2020[ca_inc_2020['LineCode'] == 2]
ca_inc_2020 = ca_inc_2020[ca_inc_2020['LineCode'] == 3]
# Rename the 2020 column to real_gdp_2020
ca_gdp_2020.rename(columns = {'2020': 'real_gdp_2020'}, inplace = True)
# Rename to 2020 column to personal_income_capita_2020
ca_inc_2020.rename(columns = {'2020': 'personal_income_capita_2020'}, inplace = True)
# Rename the 2020 column to unemployment_rate_2020
ca_unemp_2020.rename(columns = {'2020': 'unemployment_rate_2020'}, inplace = True)

ca_pop_2020.rename(columns = {'2020': 'population_2020'}, inplace = True)

# Convert 'FIPS' column to string and remove any extra characters
ca_unemp_2020['FIPS'] = ca_unemp_2020['FIPS'].astype(str).str.replace("'", '').str.strip().str.replace('.0', '').str.zfill(5)
ca_gdp_2020['GeoFIPS'] = ca_gdp_2020['GeoFIPS'].astype(str).str.replace('"', '').str.strip().str.zfill(5)
ca_inc_2020['GeoFIPS'] = ca_inc_2020['GeoFIPS'].astype(str).str.replace('"', '').str.strip().str.zfill(5)
ca_pop_2020['GeoFIPS'] = ca_pop_2020['GeoFIPS'].astype(str).str.replace('"', '').str.strip().str.zfill(5)

# Convert 'fips' column to string to ensure the keys are of the same type
covid_health_weather['fips'] = covid_health_weather['fips'].astype(str).str.replace('"', '').str.strip().str.zfill(5)

# Merge the new datasets into the main dataset using inner join
covid_health_weather = covid_health_weather.merge(ca_gdp_2020, left_on='fips', right_on='GeoFIPS', how='outer')
covid_health_weather = covid_health_weather.merge(ca_inc_2020, left_on='fips', right_on='GeoFIPS', how='outer')
covid_health_weather = covid_health_weather.merge(ca_pop_2020, left_on='fips', right_on='GeoFIPS', how='outer')
covid_health_weather = covid_health_weather.merge(ca_unemp_2020, left_on='fips', right_on='FIPS', how='outer')

# Create a gdp per capita column
covid_health_weather['real_gdp_per_capita_2020'] = covid_health_weather['real_gdp_2020'] / covid_health_weather['population_2020']

# Change date column to datetime
covid_health_weather['date'] = pd.to_datetime(covid_health_weather['date'])

# %% [markdown]
# ## Summary Statistics

# %%
# Generate summary statistics for the dataset 
summary_stats = covid_health_weather[['cases', 'mean_temp', 'real_gdp_2020', 'personal_income_capita_2020', 'dewpoint', 'wind_speed',
                                       'percent_frequent_physical_distress', 'percent_frequent_mental_distress', 'percent_no_highschool_diploma',
                                         'average_daily_pm2_5', 'high_school_graduation_rate','unemployment_rate_2020',
                                           'percent_limited_access_to_healthy_foods']].describe()
summary_stats = summary_stats.transpose()
formatted_table = tabulate(summary_stats, headers='keys', tablefmt='pretty')
print(formatted_table)


# %% [markdown]
# - Mean_temp is mean temperature. The median appears to be around 18.3 degrees celsius or 65 degrees fahrenheit, but temperature can vary between 14.5 F and 109.6 F. We will connect these varying temps to cases through analysis
# - cases counts number of covid19 cases for that city on that day. This will be our main response variable
# - real gdp in 2020 is expressed in terms of 2017 dollars. It is one of the key statistics in measuring economic wellbeing of a county and will be one our socioeconomic variables that we are connecting to covid19 cases. GDP could be negatively correlated with cases
# - We use personal income to measure income as a socioeconomic variable that can connect to number of cases. Lower income could lead to more cases
# - Dewpoint is the temperature the air needs to be cooled to reach a relative humidity of 100%. This is a measure of the moisture in the air. We will use this to predict cases
# - Percent frequent and mental distress potentially interacts with cases. We will explore how higher levels of distress impact cases. We will explore how this number interacts with the other factors
# - High school graduation rate, umemployment rate, access to healthy goods are some rate based economic variables we will explore on how they are correlated with cases

# %%
# Define thresholds for high physical and mental distress
high_physical_distress_threshold = covid_health_weather['percent_frequent_physical_distress'].quantile(0.75)
high_mental_distress_threshold = covid_health_weather['percent_frequent_mental_distress'].quantile(0.75)

# Filter the data for high physical and mental distress
high_distress_data = covid_health_weather[
    (covid_health_weather['percent_frequent_physical_distress'] >= high_physical_distress_threshold) &
    (covid_health_weather['percent_frequent_mental_distress'] >= high_mental_distress_threshold)
]

summary_stats_hd = high_distress_data[['cases', 'mean_temp', 'real_gdp_2020', 'personal_income_capita_2020', 'dewpoint', 'wind_speed',
                                       'percent_frequent_physical_distress', 'percent_frequent_mental_distress', 'percent_no_highschool_diploma',
                                         'average_daily_pm2_5', 'high_school_graduation_rate','unemployment_rate_2020',
                                           'percent_limited_access_to_healthy_foods']].describe()
summary_stats_hd = summary_stats_hd.transpose()
formatted_table_hd = tabulate(summary_stats_hd, headers='keys', tablefmt='pretty')
print(formatted_table_hd)

# %% [markdown]
# - This table is the summary statistics for the subgroup where distress is higher than the 75th quantile. I will use this subgroup to explore the effect of high distress on cases

# %% [markdown]
# ## Plots

# %%

# 1. **COVID-19 Cases Over Time**:
#     - Plot the number of COVID-19 cases over time to see the trend.

covid_health_weather.groupby('date')['cases'].sum().plot(kind='line', title='COVID-19 Cases Over Time') 
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.show()



# 2. **COVID-19 Cases vs. Mean Temperature**:
#     - Scatter plot to see the relationship between COVID-19 cases and mean temperature.
  
plt.scatter(covid_health_weather['mean_temp'], covid_health_weather['cases'])
plt.title('COVID-19 Cases vs. Mean Temperature')
plt.xlabel('Mean Temperature (°F)')
plt.ylabel('Number of Cases')
plt.show()
  
# 3. **COVID-19 Cases vs. Real GDP per Capita**:
#     - Scatter plot to see the relationship between COVID-19 cases and real GDP per capita.

plt.scatter(covid_health_weather['real_gdp_per_capita_2020'], covid_health_weather['cases'])
plt.title('COVID-19 Cases vs. Real GDP per Capita')
plt.xlabel('Real GDP per Capita (2017 dollars)')
plt.ylabel('Number of Cases')
plt.show()
    

    

# 5. **COVID-19 Cases vs. Percent Frequent Mental Distress**:
#     - Scatter plot to see the relationship between COVID-19 cases and percent frequent mental distress.

plt.scatter(covid_health_weather['percent_frequent_mental_distress'], covid_health_weather['cases'])
plt.title('COVID-19 Cases vs. Percent Frequent Mental Distress')
plt.xlabel('Percent Frequent Mental Distress')
plt.ylabel('Number of Cases')
plt.show()
    

# 6. **COVID-19 Cases vs. Unemployment Rate**:
#     - Scatter plot to see the relationship between COVID-19 cases and unemployment rate.

plt.scatter(covid_health_weather['unemployment_rate_2020'], covid_health_weather['cases'])
plt.title('COVID-19 Cases vs. Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Number of Cases')
plt.show()
    






