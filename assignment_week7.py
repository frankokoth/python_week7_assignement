#Importing the required libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_iris

#TASK 1: LOAD AND EXPLORE THE DATASET
#Loading iris dataset
iris = load_iris(as_frame=True)
df = iris.frame               #converting to pandas dataframe

df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

#Display the first few rows of the dataset using .head() to inspect data
print("First 5 rows of the dataset")
print(df.head())

#check the dataset info
print("\nDataset Info:")
print(df.info())

#check for the missing values
print("\nMissing values per column: ")
print(df.isnull().sum())


#TASK 2: BASIC DATASET ANALYSIS
# 1. Compute the basic statistics of the numerical columns (e.g., mean, median, standard deviation) using .describe().
print("\nStatistical Summary:")
print(df.describe())


# 2. Perform groupings on a categorical column (for example, species, region, or department) and compute the mean of a numerical column for each group.
grouped_lenth = df.groupby("species")["petal_length"].mean()
print("\nAverage Petal Length by Species:")
print(grouped_lenth)

# 3. Identify any patterns or interesting findings from your analysis.
print("\Findings")
print("The dataset is well balanced across species")


# TASK 3: DATA VISUALIZATION
#A. Create at least four different types of visualizations

# 1. Line chart
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal_length"], label="sepal_length", color="blue")
plt.title("Trend of Sepal Lenth over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()


#2. Bar Chart showing comparison of numerical values accross categories
plt.figure(figsize=(8,5))
plt.bar(df["species"], df["petal_length"])
# plt.bar(x="species", y="petal_length", data=df, palatte="viridis")
plt.title("Average Petal Length by Species")
plt.xlabel("species")
plt.ylabel("Average Petal Length(cm)")
plt.tight_layout()
plt.show()


#3. Histogram to show dsitribution of sepal length
plt.figure(figsize=(8,5))
plt.hist(df["sepal_length"], bins=20, color="green", edgecolor="black")
plt.title("Distribution of Sepal Lenth")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


#4 Scatter Plot showing the relationship between two numerical columns
plt.figure(figsize=(8,5))
plt.scatter(df["sepal_length"], df["petal_length"], c=df["species"].astype("category").cat.codes)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="species")
plt.tight_layout()
plt.show()


