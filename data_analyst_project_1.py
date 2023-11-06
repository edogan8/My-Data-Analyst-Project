## Importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Loading Dataset

appdata = pd.read_csv("appleAppData.csv")
head = appdata.head()
appdata.columns

#App_Id: A unique identifier for each app in the dataset.
#App_Name: The name of the app.
#AppStore_Url: The URL of the app in the app store.
#Primary_Genre: The primary genre of the app.
#Content_Rating: The content rating that describe the minimum maturity level of content in apps.
#Size_Bytes: The size of the app in bytes.
#Required_IOS_Version: The minimum version of iOS required to run the app.
#Released: The date when the app was released.
#Updated: The date when the app was last updated.
#Version: The version number of the app.
#Price: The price of the app.
#Currency: The currency in which the price is listed.
#Free: A boolean value indicating whether the app is free or paid.
#DeveloperId: A unique identifier for the developer of the app.
#Developer: The name of the developer of the app.
#Developer_Url: The URL of the developer's page in the app store.
#Developer_Website: The website of the app developer.
#Average_User_Rating: The average user rating of the app.
#Reviews: The user feedbacks or reviews, to indicate its quality and user satisfaction for app.
#Current_Version_Score: The current version score of the app.
#Current_Version_Reviews: The number of reviews for the current version of the app.

# If the column names contained spaces, we would have to convert them to the "_" symbol.
# appdata.columns = appdata.columns.str.replace(" ","_")

## Understanding the data

appdata.shape
appdata.dtypes
appdata.describe().T
appdata.info()

## Lets see the missing values

appdata.isnull().sum() 
# The most missing observations are in the "Developer_Website" column.
# We can leave this column out. We foresaw that it was not meaningful.

# Graph for missing observations

sns.set_theme()
sns.set(rc={"figure.dpi":300,
            "figure.figsize":(12,9)})


sns.heatmap(appdata.isnull(),cbar=False)
plt.title("Missing null values")

# Cleaning Data

appdata.drop(columns="Developer_Website", inplace=True)

appdata.isnull().sum() 
# We looked again and "App_Name" and
# We have decided to delete a small number of missing observations found in the "Released" columns from the data.
# Because it would be unreasonable to fill in these missing observations, and at the same time it would be unnecessary
# to delete the entire column.

appdata.dropna(subset=['App_Name'], inplace=True)
appdata.dropna(subset=['Released'], inplace=True)

appdata.isnull().sum()

appdata["Size_Bytes"] = appdata["Size_Bytes"].fillna(appdata["Size_Bytes"].median())

appdata["Price"] = appdata["Price"].fillna(appdata["Size_Bytes"].median())

appdata.isnull().sum()

appdata["Developer_Url"].fillna(0, inplace=True)

appdata.isnull().sum()

# Looking again the Dataset

appdata.nunique()

# Let's look at repeated observations.

appdata.duplicated().any()

appdata[["App_Id"]].duplicated().sum()
appdata[["App_Name"]].duplicated().sum()

# Let's look at the repeated "App_Name" observations.

appdata["App_Name"].value_counts().sort_values()

# A problem since we know that there are no duplicate observations in the "App_Id" column, 
# even if we see that there are repeated names in the "App_Name" column 
# we foresaw that it did not constitute.

appdata.columns

appdata["Content_Rating"].unique()

# We understood that the "Content_Rating" column determines the age.

appdata['Content_Rating'].value_counts().sort_values()

appdata["Age_Group"] = appdata["Content_Rating"].map({"4+":"The Firs Childhood",
                                                           "9+":"The Second Childhood",
                                                           "12+":"Teens",
                                                           "17+":"Adults",
                                                           "Not yet rated":"Everyone"})

appdata[["Content_Rating","Age_Group"]].head(50)

appdata["Age_Group"].value_counts()

# The "Size_Bytes" column seems to be quite confusing.
# We need to do MB conversion to better understand it.
# MB : megabytes = bytes/1024/1024

appdata["Size_Bytes"].dtypes
appdata["Size_MB"] = appdata["Size_Bytes"].apply(lambda x: (float(x)/1024)/1024)

appdata["Size_MB"].head()

appdata.columns

# Let's take the "Free" column now.

appdata["Free"].unique()
# Here, True:Free, False:Paid means.
# Now let's fix this.

appdata["Type"] = np.where(appdata["Free"]==True,
                           "Free", "Paid")

appdata["Type"].head()

# Take the "Reviews" column.This column is important for us
# because the number of views will be useful for us in terms of visualization.

appdata["Reviews"].describe().T
appdata["Reviews"].unique()

appdata['ReviewCategory'] = ""
appdata.loc[appdata["Reviews"] <= 10000, "ReviewCategory"] = "Less_than_10K"
appdata.loc[(appdata["Reviews"] > 10000) & (appdata["Reviews"]<= 500000),"ReviewCategory"] = "10K_to_500K"
appdata.loc[(appdata["Reviews"] > 500000) & (appdata["Reviews"]<= 1000000),"ReviewCategory"] = "500K_to_1000k"
appdata.loc[(appdata["Reviews"] > 1000000) & (appdata["Reviews"]<= 22685334),"ReviewCategory"] = "Million_Plus"
appdata[["Reviews","ReviewCategory"]].head()

appdata["ReviewCategory"].value_counts()

head2 = appdata.head(500)

appdata["Reviews"].max()

appdata["App_Name"].loc[appdata["Reviews"]==22685334]

top_reviews_app = appdata[["App_Name", "ReviewCategory", "Reviews"]].loc[
    appdata["Reviews"] >= 1000000].sort_values(by="Reviews",ascending=False).reset_index(drop=True)

top_reviews_app.head(50)

# Take the "Average_User_Rating" column.

appdata["Average_User_Rating"].unique()

appdata["Average_User_Rating"] = appdata["Average_User_Rating"].round().astype(int)

appdata["Average_User_Rating"].head(10)
appdata["Average_User_Rating"].unique()

# Let's look at the "Released" column.

appdata["Released"].unique()
appdata["Released"].dtype

appdata["Release_Year"] = pd.to_datetime(appdata["Released"]).dt.strftime("%Y")

# Let's look at the "Updated" column.

appdata["Updated"].unique()
appdata["Updated"].dtype

appdata["Updated_Year"] = pd.to_datetime(appdata["Updated"]).dt.strftime("%Y")

head3 = appdata.head()

# Now let's take a look at the prices of these applications.

appdata["Price"].min()

appdata["Price"].max()

appdata["PriceRange"] = ""
appdata.loc[appdata["Price"] == 0, "PriceRange"] = "Free"
appdata.loc[(appdata["Price"]>0.1) & (appdata["Price"]<=1), "PriceRange"] = "0_1"
appdata.loc[(appdata["Price"]>1) & (appdata["Price"]<=50), "PriceRange"] = "1_50"
appdata.loc[(appdata["Price"]>50) & (appdata["Price"]<=100), "PriceRange"] = "50_100"
appdata.loc[(appdata["Price"]>100) & (appdata["Price"]<=200), "PriceRange"] = "100_200"
appdata.loc[(appdata["Price"]>200) & (appdata["Price"]<=300), "PriceRange"] = "200_300"
appdata.loc[(appdata["Price"]>300) & (appdata["Price"]<=400), "PriceRange"] = "300_400"
appdata.loc[(appdata["Price"]>400) & (appdata["Price"]<=500), "PriceRange"] = "400_500"
appdata.loc[(appdata["Price"]>500) & (appdata["Price"]<=1000), "PriceRange"] = "500_1000"
appdata.loc[appdata["Price"]>1000, "PriceRange"] = "1000+"
appdata[["Price", "PriceRange"]].head(100)

appdata["PriceRange"].value_counts()

head4 = appdata.head(500)

## Data Visualization

# Let's look at the number of paid and free applications.

appdata["Type"].value_counts().plot(kind="bar",
                                    color="red") 


# Let's look at the "Content_Rating" column.

sns.countplot(y = "Age_Group", data=appdata)
plt.title("Content Rating with their counts")


# Let's look at the number of applications by category.

cat_num = appdata["Primary_Genre"].value_counts()
sns.barplot(x = cat_num, y = cat_num.index, data=appdata)
plt.title("The number of categories", size=20)  

# Lets see the top 10 reviews apps

top_reviews_app = appdata[["App_Name", "ReviewCategory", "Reviews"]].loc[
    appdata["Reviews"] >= 1000000].sort_values(by="Reviews",ascending=False).reset_index(drop=True).head(10)
top_reviews_app

plt.figure(figsize=(20,8))
sns.barplot(data=top_reviews_app, x="App_Name", 
            y="Reviews")
plt.xticks(rotation=90)

# Lets see Released_Year vs each years.

app_counts_per_year = appdata["Release_Year"].value_counts().reset_index(name="Released_apps").rename(columns=
                                                                                                      {"index":"Year",
                                                                                                       "Release_Year":"Released_apps"})
app_counts_per_year

plt.figure(figsize=(20,8))
sns.barplot(data=app_counts_per_year, x="Year",
            y="Released_apps")
plt.title("App counts per year", fontdict={"fontsize":30})
plt.xlabel("Years", fontdict={"fontsize":30})
plt.ylabel("Released app count", fontdict={"fontsize":30})

# Lets see top 50 Education apps

top_edu_apps = appdata[["App_Name", "Primary_Genre", "Reviews"]].loc[appdata[
    "Primary_Genre"]=="Education"].sort_values(by="Reviews",ascending=False).reset_index(drop=
                                                                                        "True").head(50)
top_edu_apps

plt.figure(figsize=(20,8))
sns.barplot(data=top_edu_apps, x="App_Name", y=
            "Reviews")
plt.title("Top50 Apps for Reviews")
plt.xlabel("App Name")
plt.ylabel("Reviews")
plt.xticks(rotation=90)

# Which developer has the most apps in the AppleApps Store?

top_10_app_dev = appdata.groupby(["Developer"]).size().sort_values(ascending=
                                                              False).reset_index(name="Sum").head(10)
top_10_app_dev

plt.figure(figsize=(20,8))
data = top_10_app_dev.groupby("Developer")["Sum"].max().sort_values()
labels = data.index.tolist()
plt.pie(data, labels=labels, autopct="%.0f%%")

# What are the categories with the highest number of paid apps?

most_paid = appdata[["Primary_Genre","Type"]].loc[appdata["Type"]==
"Paid"].value_counts(ascending=False).reset_index(name="Sum").head(10)
most_paid

# What are the categories with the highest number of free apps?

most_free = appdata[["Primary_Genre","Type"]].loc[appdata["Type"]==
"Free"].value_counts(ascending=False).reset_index(name="Sum").head(10)
most_free

# How are the numeric attributes correlated with each other?

sns.heatmap(appdata.corr(), annot=True, 
            linewidths=.5, fmt=".2f")

