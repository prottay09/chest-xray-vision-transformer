from model.ViT import ViT
from pathlib import Path
from datamodule.ChexpertDataModule import CheXpertDataModule
from sklearn import metrics
import matplotlib.pyplot as plt
from model.Vit_pretrained import pretrained_vit
from model.ResNet50 import Resnet50
from model.deit import DeiT
from model.crossViT import Cross_ViT
from model.deepvit import deep_ViT
PATH = r'C:/Users/prott/OneDrive/Desktop/Thesis/cheXpert/chexpert-lightning-prottay/lightning_logs/version_64/checkpoints/last.ckpt'

from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
import timeit
    


model = pretrained_vit.load_from_checkpoint(PATH)

# print(model.learning_rate)
model.eval()

img_path = 'view1_frontal.jpg'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

#y_true, y_pred = model(CheXpertData)
im = Image.open(img_path).convert('RGB')
x = transform(im)
x = x.unsqueeze(0)
logits, att_mat = model(x)
att_mat = torch.stack(att_mat).squeeze(1)

# Average the attention weights across all heads.
att_mat = torch.mean(att_mat, dim=1)

# To account for residual connections, we add an identity matrix to the
# attention matrix and re-normalize the weights.
residual_att = torch.eye(att_mat.size(1))
aug_att_mat = att_mat + residual_att
aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

# Recursively multiply the weight matrices
joint_attentions = torch.zeros(aug_att_mat.size())
joint_attentions[0] = aug_att_mat[0]

for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    
# Attention from the output token to the input space.
v = joint_attentions[-1]
grid_size = int(np.sqrt(aug_att_mat.size(-1)))
mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
result = (mask * im).astype("uint8")
print(logits)
result_list = []
for i, v in enumerate(joint_attentions):
    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")
    result_list.append(result)
    # plt.figure(figsize=(20,20))
    # plt.title('Attention Map_%d Layer' % (i+1), fontsize=20)
    # plt.imshow(result)
    # #plt.show()
    # #plt.show()
    # plt.savefig(f'Attention_with_weight_{i+1}.jpg',edgecolor = "white", facecolor='white', transparent=False, frameon = True)

fig, ax = plt.subplots(nrows=6, ncols=2, figsize = (60,120))
i = 0
for row in ax:
    for col in row:
        col.imshow(result_list[i])
        col.set_title('Attention Map_%d Layer' % (i+1), fontsize=50)
        i = i + 1
plt.savefig(f'Attention_with_weight.jpg',edgecolor = "white", facecolor='white', transparent=False, frameon = True)