import pytorch_lightning as pl
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module.Layers import *
from sklearn import metrics
import matplotlib.pyplot as plt
import json
from torchvision.models import resnet50
from vit_pytorch.distill import DistillableViT, DistillWrapper

class DeiT(pl.LightningModule):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DeiT, self).__init__()
        self.save_hyperparameters()
        self.teacher = resnet50(pretrained=True)
        num_ftrs = self.teacher.fc.in_features
        self.teacher.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

        self.student = DistillableViT(
                        image_size = 224,
                        patch_size = 16,
                        num_classes = 5,
                        dim = 1024,
                        depth = 6,
                        heads = 8,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1
                    )

        self.distiller = DistillWrapper(
                            student = self.student,
                            teacher = self.teacher,
                            temperature = 3,           # temperature of distillation
                            alpha = 0.5,               # trade between main loss and distillation loss
                            hard = True               # whether to use soft or hard distillation
                        )
        #self.loss = torch.nn.BCELoss(size_average = True)

    def forward(self, x):
        x = self.student(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam (filter(lambda p: p.requires_grad, self.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        varInput, varTarget = batch
        #varOutput = self(varInput)
        lossvalue = self.distiller(varInput, varTarget)
        varOutput = torch.sigmoid(self.student(varInput))
        #lossvalue = self.loss(varOutput, varTarget)
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

        plt.savefig("DeiT_w_100_soft_scratch.png", dpi=1000)
        plt.show()

        #store the result
        with open("result_100_deit_w_hard.json", "w") as outfile:
            # result = json.load(outfile)
            # result.update(json.loads(json.dumps(result_dict)))
            # outfile.seek(0)
            json.dump(result_dict, outfile)
        return {'test_loss' : avg_test_loss}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        return {'val_loss' : avg_val_loss}

