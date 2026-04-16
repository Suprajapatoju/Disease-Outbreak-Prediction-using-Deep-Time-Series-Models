import pandas as pd

df = pd.read_csv("Final_data_large.csv", encoding="latin1")

# remove extra spaces
df["district"] = df["district"].str.strip()

# standardize names
df["district"] = df["district"].replace({
    "Chittor": "Chittoor",
    "Godavari": "East Godavari"
})

# make consistent format
df["district"] = df["district"].str.title()

# save cleaned dataset
df.to_csv("Final_data_clean.csv", index=False)

print("Dataset cleaned successfully")
print("New shape:", df.shape)