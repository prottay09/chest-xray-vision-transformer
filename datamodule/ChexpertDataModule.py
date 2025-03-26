from utils.ChexpertDataset import CheXpertDataSet
import pytorch_lightning as pl
import torchvision.transforms as transforms
from typing import Optional
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, pathFileTrain : str, pathFileValid : str, pathFileTest : str):
        super().__init__()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.pathFileTrain = pathFileTrain
        self.pathFileTest = pathFileTest
        self.pathFileValid = pathFileValid
        self.imgtransCrop = 224
        self.imgtransResize = (224, 224)
        self.trBatchSize = 16
        self.transformSequenceTrain = transforms.Compose([transforms.ToTensor(), self.normalize, transforms.RandomResizedCrop(self.imgtransCrop), transforms.RandomHorizontalFlip()])
        self.transformSequenceTest = transforms.Compose([transforms.ToTensor(), self.normalize, transforms.Resize(self.imgtransResize)])
    def setup(self, stage: Optional[str] = None):
        
        if stage in (None, "fit"):
            self.datasetTrain = CheXpertDataSet(self.pathFileTrain, self.transformSequenceTrain, policy="ones")
            self.datasetValid = CheXpertDataSet(self.pathFileTrain, self.transformSequenceTest, policy="ones")
        if stage in (None, "test"):
            self.datasetTest = CheXpertDataSet(self.pathFileTest, self.transformSequenceTest, policy="ones")            
    
    def train_dataloader(self):
        return DataLoader(dataset=self.datasetTrain, batch_size=self.trBatchSize, shuffle=True,  num_workers=1, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.datasetValid, batch_size=self.trBatchSize, shuffle=False, num_workers=1, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.datasetTest, batch_size=self.trBatchSize, shuffle=False, num_workers=1, pin_memory=True)