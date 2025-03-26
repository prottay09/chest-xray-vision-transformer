#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
# %%
path = "C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small"
train_path = "C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/train.csv"
# %%
train_df = pd.read_csv(train_path)
#train_df = train_df.fillna('NaN')

# %%
# select top 5 based on clinical importance and prevalence
top_5 = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
# %%

train_df = train_df[train_df['AP/PA']=='AP']
train_df = train_df[top_5]
train_df = train_df.replace(-1,1)
train_df.fillna(0, inplace=True)
# %%
for cat in train_df.columns:
    print(cat)
    print(train_df[cat].value_counts())

# %%
import matplotlib.pyplot as plt
sns.set_palette('pastel')
fig = dist.plot(kind='bar', stacked = True, rot = 0, figsize=(20,6), fontsize = 20)
plt.savefig('distribution2.png',edgecolor = "white", facecolor='white', transparent=False, frameon = True)
#%%
dist = pd.DataFrame(data={'NaN': [154971, 177211, 152792, 137458, 90203], 'Positive': [33376, 27000, 14783, 52246, 86187], 'Negative': [1328, 11116, 28097, 20726, 35396], 'Uncertain': [33739, 8087, 27742, 12984, 11628]}, index = top_5)
# %%
# for Atelectasis and Cardiomegaly 33739 and 27742 samples are uncertain respectively
# so, setting any unceratinty policy will severely effect on the labelling
# what does it mean by U-ignore
# %%
def apply_filter(df):
    cols = ['Path','Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'patient']
    df = df[df['AP/PA']=='AP']
    df = df[cols]
    #df.dropna(thresh=3, inplace=True) # to remove samples with no positive
    df.fillna(0, inplace=True)
    return df
#%%
# load finn's data splits
new_train_path = "C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/Chex_train.csv"
new_train_df = pd.read_csv(new_train_path)
new_train_df = apply_filter(new_train_df)
new_train_df = new_train_df.replace(-1,1)
for cat in new_train_df.columns:
    print(cat)
    print(new_train_df[cat].value_counts())
#%%
pos_weights = []
#%%
# splitting the data into 30, 60 and 100
index = list(new_train_df.groupby('patient').count().sort_values(by='Path', ascending=False).index)
len_30 = len(index)*3//4
len_60 = len(index)*15//16
train_30 = new_train_df[new_train_df['patient'].isin(index[-len_30:])]
train_60 = new_train_df[new_train_df['patient'].isin(index[-len_60:])]
train_60.to_csv("C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/train_60.csv")
train_30.to_csv("C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/train_30.csv")
# %%
new_val_path = "C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/Chex_val.csv"
new_val_df = pd.read_csv(new_val_path)
new_val_df = apply_filter(new_val_df)
msk = np.random.rand(len(new_val_df))<0.67
val_60 = new_val_df[msk]
msk = np.random.rand(len(val_60))<0.5
val_30 = val_60[msk]
val_60.to_csv("C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/val_60.csv")
val_30.to_csv("C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/val_30.csv")
# %%
# in Finn's split Atelectasis and Consolidation were the worst
new_test_path = "C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/Chex_test.csv"
new_test_df = pd.read_csv(new_test_path)
new_test_df = apply_filter(new_test_df)
msk = np.random.rand(len(new_test_df))<0.67
test_60 = new_test_df[msk]
msk = np.random.rand(len(test_60))<0.5
test_30 = test_60[msk]
test_60.to_csv("C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/test_60.csv")
test_30.to_csv("C:/Users/prott/OneDrive/Desktop/Thesis/CheXpert-v1.0-small/test_30.csv")
# %%
for cat in new_test_df.columns:
    print(cat)
    print(new_test_df[cat].value_counts())
# %%
