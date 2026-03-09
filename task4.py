import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Step 1: Load dataset (no headers in file, so use header=None)
url = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%204/twitter_training.csv"
data = pd.read_csv(url, header=None)

# Step 2: Assign column names
data.columns = ['id', 'entity', 'sentiment', 'tweet']

print("Dataset shape:", data.shape)
print("First 5 rows:")
print(data.head())

# Step 3: Data cleaning
# Drop duplicates
data.drop_duplicates(inplace=True)

# Define cleaning function
def clean_text(text):
    text = str(text).lower()                     # convert to string + lowercase
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)         # remove punctuation/numbers
    return text

# Apply cleaning function
data['clean_tweet'] = data['tweet'].apply(clean_text)

# Verify cleaning
print(data[['tweet', 'clean_tweet']].head())

# Step 4: Exploratory Data Analysis (EDA)

# Sentiment distribution
sns.countplot(x='sentiment', data=data)
plt.title("Sentiment Distribution")
plt.show()

# Tweet length distribution
data['tweet_length'] = data['tweet'].apply(lambda x: len(str(x)))
sns.histplot(data['tweet_length'], bins=30, kde=True)
plt.title("Tweet Length Distribution")
plt.show()