import pytorch_lightning as pl
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim

from sklearn import metrics
import matplotlib.pyplot as plt
import json

import io
import os
import sys
print(os.getcwd())

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms
import typing
from models.modeling import VisionTransformer, CONFIGS
os.makedirs("attention_data", exist_ok=True)
if not os.path.isfile("attention_data/ilsvrc2012_wordnet_lemmas.txt"):
    urlretrieve("https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt", "attention_data/ilsvrc2012_wordnet_lemmas.txt")
if not os.path.isfile("attention_data/ViT-B_16-224.npz"):
    urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz", "attention_data/ViT-B_16-224.npz")

#%%
class pretrained_vit(pl.LightningModule):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(pretrained_vit, self).__init__()
        self.save_hyperparameters()
        config = CONFIGS["ViT-B_16"]
        self.vit = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
        self.vit.load_from(np.load("attention_data/ViT-B_16-224.npz"))
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.FloatTensor([2.097, 4.91, 3.87, 1.64, 1]))

    def forward(self, x):
        x, attn_weights = self.vit(x)
        return x, attn_weights
    
    def configure_optimizers(self):
        optimizer = optim.Adam (self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        varInput, varTarget = batch
        varOutput = self(varInput)[0]
        lossvalue = self.loss(varOutput, varTarget)
        varOutput = torch.sigmoid(varOutput)
        
        return {'loss': lossvalue, 'preds': varOutput, 'targets': varTarget} 

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        return {'val_loss' : avg_val_loss}

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        avg_test_loss = torch.tensor([x['loss'] for x in test_step_outputs]).mean()
        print(f'Average test loss is {avg_test_loss}')
        preds = torch.cat([x['preds'] for x in test_step_outputs]).cpu()
        targets = torch.cat([x['targets'] for x in test_step_outputs]).cpu()
        pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        result_dict = {}
        for i, patho in enumerate(pathologies):
            result_dict[pathologies[i]] = {}
            target = targets[:,i]
            pred = preds[:,i]
            fpr, tpr, threshold = metrics.roc_curve(target, pred)
            roc_auc = metrics.auc(fpr, tpr)
            result_dict[pathologies[i]]['fpr'] = fpr.tolist()
            result_dict[pathologies[i]]['tpr'] = tpr.tolist()
            result_dict[pathologies[i]]['target'] = target.tolist()
            result_dict[pathologies[i]]['pred'] = pred.tolist()
            # plotting
            plt.subplot(5, 1, i+1)
            plt.title('ROC for 100%: ' + pathologies[i])
            plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
        
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 30
        fig_size[1] = 10
        plt.rcParams["figure.figsize"] = fig_size

        plt.savefig("nothing.png", dpi=1000)
        plt.show()

        #store the result
        with open("nothing.json", "w") as outfile:
            # result = json.load(outfile)
            # result.update(json.loads(json.dumps(result_dict)))
            # outfile.seek(0)
            json.dump(result_dict, outfile)
        return {'test_loss' : avg_test_loss}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        return {'val_loss' : avg_val_loss}

