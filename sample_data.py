import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold, GroupKFold

path_data_train = r'C:\Users\Finn\Documents\Projects\LUMEN\Code\Pytorch Lighning Lumen\data\CheXpert-v1.0-small\train.csv'
path_data_val = r'C:\Users\Finn\Documents\Projects\LUMEN\Code\Pytorch Lighning Lumen\data\CheXpert-v1.0-small\valid.csv'
imgpath = r'C:\Users\Finn\Documents\Projects\LUMEN\Code\Pytorch Lighning Lumen\data\CheXpert-v1.0-small/'
savePath =  r'C:\Users\Finn\Documents\Projects\LUMEN\Code\Pytorch Lighning Lumen\data/splits/'

# basepath_plots = r'C:\Users\Finn\Documents\Projects\LUMEN\Dataset investigation\plots\Chexpert/'
df = pd.read_csv(path_data_train)
df_val = pd.read_csv(path_data_val)
pathologies = ["No Finding",
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Lung Opacity",
                "Lung Lesion",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
                "Pleural Other",
                "Fracture",
                "Support Devices"]

# pathologies = [ 
#                 # 'No Finding',
#                 # 'Enlarged Cardiomediastinum', 
#                 'Cardiomegaly', 
#                 # 'Lung Opacity',
#                 # 'Lung Lesion', 
#                 'Edema', 
#                 'Consolidation', 
#                 # 'Pneumonia', 
#                 'Atelectasis',
#                 # 'Pneumothorax', 
#                 'Pleural Effusion' 
#                 # 'Pleural Other', 
#                 # 'Fracture'
#                 # 'Support Devices'
#                 ]
# X = df
# X_train, X_test = train_test_split(X, test_size=0.33, random_state=42,stratify=df[pathologies])

def save_pichart_classes(df,name=None):
    means = {}
    cases = pd.DataFrame()

    for pathology in pathologies:
        means[pathology] = df.Age[df[pathology]==1].mean()
        cases[pathology] = df[pathology][df[pathology]==1].value_counts()/len(df)

    print(len(df[df['Frontal/Lateral']=='Frontal'])/len(df))
    print(len(df[df['Frontal/Lateral']=='Lateral'])/len(df))
    # print(means)
    # cases.T.plot.pie(subplots=True,labeldistance=None,autopct=None)
    # cases.plot.bar()
    print(cases)
    # plt.legend(bbox_to_anchor=(1, 0.4, 0.1, 0.5))
    # plt.savefig(basepath_plots+'classes_piechart_{}.png'.format(name), bbox_inches='tight')
    # plt.show()
df['patient'] = df.Path.str.split("train/patient", expand=True)[1].str.split('/study',expand=True)[0] # For Grouping

df_replaced = df.replace(-1.0, 1)
df_replaced = df_replaced.fillna(0)
# Split to train/val and Test Data with group awareness and View stratification
cv = StratifiedGroupKFold(n_splits=5,shuffle = True, random_state=42)
for fold, (train_inds, test_inds) in enumerate(cv.split(X=df_replaced, y=df_replaced['Frontal/Lateral'], groups=df_replaced['patient'])):
# train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 42).split(df, groups=df['patient']))
    if fold == 0:
        train_df = df.iloc[train_inds]
        test_df = df.iloc[test_inds]
save_pichart_classes(train_df)
save_pichart_classes(test_df)
test_df.to_csv(savePath+f'chex_test.csv')
# Split training set into n folds for crossvalidation
# cv = GroupKFold(n_splits=5)

df_replaced = train_df.replace(-1.0, 1)
df_replaced = df_replaced.fillna(0)


for fold , (train_inds, test_inds) in enumerate(cv.split(X=df_replaced, y=df_replaced['Frontal/Lateral'], groups=df_replaced['patient'])):
    train_df_cv = df.iloc[train_inds]
    val_df_cv = df.iloc[test_inds]
    save_pichart_classes(train_df_cv)
    save_pichart_classes(val_df_cv)
    train_df_cv.to_csv(savePath+f'chex_train_fold{fold}.csv')
    val_df_cv.to_csv(savePath+f'chex_val_fold{fold}.csv')



print(f'Length of Training Set(s): {len(train_df_cv)}')
print(f'Length of Validation Set(s): {len(val_df_cv)}')
print(f'Length of Test Set: {len(test_df)}')
print('done')