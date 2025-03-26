import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd



# dataset should contain three classes - __init__, __getitem__, __len__
class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file, transform=None, policy="ones"):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        image_names = []
        labels = []

        # with open(image_list_file, "r") as f:
            
        #     csvReader = csv.reader(f)
        #     next(csvReader, None)
        #     k=0
        #     for line in csvReader:
        #         k+=1
        #         image_name= line[0]
        #         label = line[5:]
                
        #         for i in range(14):
        #             if label[i]:
        #                 a = float(label[i])
        #                 if a == 1:
        #                     label[i] = 1
        #                 elif a == -1:
        #                     if policy == "ones":
        #                         label[i] = 1
        #                     elif policy == "zeroes":
        #                         label[i] = 0
        #                     else:
        #                         label[i] = 0
        #                 else:
        #                     label[i] = 0
        #             else:
        #                 label[i] = 0
                        
        #         image_names.append(Path(__file__).parents[3].joinpath(image_name))
        #         labels.append(label)
        # select top 5 based on clinical importance and prevalence
        cols = ['Path','Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

        df = pd.read_csv(image_list_file)
        #df = df[df['AP/PA']=='AP']
        df = df[cols]
        #df.dropna(thresh=2, inplace=True)
        df.fillna(0, inplace=True)

        if policy == "ones":
            df = df.replace(-1,1)
        elif policy == "zeros":
            df = df.replace(-1,0)

        for _ , row in df.iterrows():
            image_names.append(Path(__file__).parents[3].joinpath(row[0]))
            labels.append(list(row[1:].values))





        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)