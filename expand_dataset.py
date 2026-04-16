import pandas as pd
import numpy as np

df = pd.read_csv("Final_data.csv")

# convert numeric columns
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
df["preci"] = pd.to_numeric(df["preci"], errors="coerce")
df["LAI"] = pd.to_numeric(df["LAI"], errors="coerce")

df = df.dropna()

augmented = []

for i in range(4):

    temp_df = df.copy()

    temp_df["Temp"] = temp_df["Temp"] + np.random.normal(0,1,len(temp_df))
    temp_df["preci"] = temp_df["preci"] + np.random.normal(0,2,len(temp_df))
    temp_df["LAI"] = temp_df["LAI"] + np.random.normal(0,0.05,len(temp_df))

    temp_df["Cases"] = (temp_df["Cases"] + np.random.randint(-3,4,len(temp_df))).clip(lower=0)

    temp_df["day"] = (temp_df["day"] + np.random.randint(0,3,len(temp_df))).clip(1,28)

    augmented.append(temp_df)

df_big = pd.concat([df] + augmented, ignore_index=True)

df_big.to_csv("Final_data_large.csv", index=False)

print("Original size:", len(df))
print("New dataset size:", len(df_big))