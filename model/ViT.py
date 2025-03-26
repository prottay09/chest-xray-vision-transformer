import pytorch_lightning as pl
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
from einops import rearrange, repeat
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, AUROC, AveragePrecision, F1, AUC, PrecisionRecallCurve
import sklearn.metrics as metrics
from einops.layers.torch import Rearrange
from module.Layers import *
import matplotlib.pyplot as plt
import json


class ViT(pl.LightningModule):
    def __init__(self, img_size: int = 256, patch_size: int = 16, 
                num_class: int = 14, d_model: int = 768, n_head: int = 12, 
                n_layers:int = 12, d_mlp: int = 3072, channels: int = 3, 
                dropout: float = 0., pool: str = 'cls'):
        super().__init__()
        self.save_hyperparameters()

        img_h, img_w = img_size, img_size
        patch_h, patch_w = patch_size, patch_size

        assert img_h % patch_h == 0, 'image dimension must be divisible by patch dimension'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        num_patches = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        self.patches_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_h, p2 = patch_w),
            nn.Linear(patch_dim, d_model)
        )

        self.pos_embed = PositionalEncoding(d_model, num_patches, dropout)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool = pool
    
        self.transformer = Transformer(d_model, n_head, n_layers, d_mlp, dropout)
        self.dropout = nn.Dropout(dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_class),
            nn.Sigmoid()
        )
        self.loss = torch.nn.BCELoss()

    def forward(self, img):
        x = self.patches_embed(img)
        b, n, _ = x.shape
        class_token = repeat(self.class_token, '() n d -> b n d', b = b)
        #Concat Class Token with image patches
        x = torch.cat((class_token,x), dim=1)
        #Add Positional Encoding
        x = self.pos_embed(x, n)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #MLP Head
        x = self.mlp_head(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam (self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        varInput, varTarget = batch
        varOutput = self(varInput)
        lossvalue = self.loss(varOutput, varTarget)
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

        plt.savefig("ViT_100.png", dpi=1000)
        plt.show()

        #store the result
        with open("result_100_ViT.json", "w") as outfile:
            # result = json.load(outfile)
            # result.update(json.loads(json.dumps(result_dict)))
            # outfile.seek(0)
            json.dump(result_dict, outfile)
        return {'test_loss' : avg_test_loss}
    
class Transformer(pl.LightningModule):
    def __init__(self, d_model: int = 768, n_head: int = 12, n_layers:int = 12,
                d_mlp: int = 3072, dropout: float = 0.):
        super().__init__()

        self.block = nn.ModuleList([
            Norm(d_model, MultiHeadAttention(d_model, n_head, dropout)),
            Norm(d_model, FeedForward(d_model, d_mlp, dropout))
            ])
        self.layers = nn.ModuleList([self.block for _ in range(n_layers)])

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x) + x
            x = mlp(x) + x
        return x
