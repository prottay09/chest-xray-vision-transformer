#%%
import json
from pathlib import Path
import glob
from tkinter import font
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from sklearn.metrics import auc, f1_score, cohen_kappa_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('Paired')
pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
#%%
result_dic = {}
for file in glob.glob('*.json'):
    x = file.split('.')[0]
    with open(Path(file), "r") as f:
        result_dic[x] = json.load(f)

# %%

for key in ['result_100_CrossViT2_scratch', 'result_100_CrossViT2_scratch_w', 'result_100_CrossViT_scratch']:
    print(key)
    f1_scores = 0
    cohen_kappa_scores = 0
    # calculate AUROC for each pathology
    for i in pathologies:
        fpr = result_dic[key][i]['fpr']
        tpr = result_dic[key][i]['tpr']
        rocauc = auc(fpr, tpr)
        print(f'{key}_{i}_rocauc_{rocauc}')
        y_true = np.array(result_dic[key][i]['target'])
        y_pred = 1*(np.array(result_dic[key][i]['pred'])>0.5)
        f1 = f1_score(y_true, y_pred)
        f1_scores+=f1
        #print(f'{key}_{i}_F1_score_{f1}')
        cks = cohen_kappa_score(y_true, y_pred)
        cohen_kappa_scores+=cks
        #print(f'{key}_{i}_cohen-kappa_score_{cks}')
    print(f'Average F1 score for {key} is {f1_scores/5}')
    print(f'Average Cohen kappa score for {key} is {cohen_kappa_scores/5}')


#%%
keys = ['result_30_resnet50', 'result_30_ViT', 'result_60_resnet50', 'result_60_ViT', 'result_100_resnet50', 'result_100_ViT']
result_dic1 = {x:result_dic[x] for x in keys}
# %%

#%%
# Edema
result = pd.DataFrame()
for key in result_dic1:
    fpr = result_dic[key]['Edema']['fpr']
    tpr = result_dic[key]['Edema']['tpr']
    rocauc = auc(fpr, tpr)
    if '30' in key:
        data = '30%'
    elif '60' in key:
        data = '60%'
    else:
        data = '100%'
    df = pd.DataFrame([{'rocauc' : rocauc, 'model' : 'ResNet' if 'resnet' in key else 'ViT', 'Data' : data}])
    result = pd.concat([result, df], axis = 0)
result.index = np.array([1,2,3,4,5,6])
# result = result.reindex([1,2,5,6,3,4])
sns.set(font_scale = 1.4)
plt.figure(figsize=(10,10))
sns.barplot(data=result, x = 'model', y='rocauc', hue='Data')
plt.title('Different ROCAUC scores for Edema', fontsize = 20)
plt.savefig("ROCAUC_scores_different_models.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
#%%
result = pd.DataFrame()
for key in result_dic:
    y_true = np.array(result_dic[key]['Edema']['target'])
    y_pred = 1*(np.array(result_dic[key]['Edema']['pred'])>0.5)
    
    f1 = f1_score(y_true, y_pred)
    
    if '30' in key:
        data = '30%'
    elif '60' in key:
        data = '60%'
    else:
        data = '100%'
    df = pd.DataFrame([{'F1' : f1, 'model' : 'DenseNet' if 'dense' in key else 'ViT', 'Data' : data}])
    result = pd.concat([result, df], axis = 0)
result.index = np.array([1,2,3,4,5,6])
result = result.reindex([1,2,5,6,3,4])
sns.set(font_scale = 1.4)
plt.figure(figsize=(10,10))
sns.barplot(data=result, x = 'model', y='F1', hue='Data')
plt.title('Different F1 scores for Edema', fontsize = 20)
plt.savefig("F1_scores_different_models.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
#%%
# Dense vs ViT
for i in pathologies:
    fpr1 = result_dic['result_100_dense'][i]['fpr']
    tpr1 = result_dic['result_100_dense'][i]['tpr']
    fpr2 = result_dic['result_100_ViT'][i]['fpr']
    tpr2 = result_dic['result_100_ViT'][i]['tpr']
    rocauc_1 = auc(fpr1, tpr1)
    rocauc_2 = auc(fpr2, tpr2)
    plt.figure(figsize=(10,10))
    plt.title(f'{i}', fontsize = 20)
    plt.plot(fpr1, tpr1, label = f'ROCAUC_Densenet = {rocauc_1:.2f}')
    plt.plot(fpr2, tpr2, label = f'ROCAUC_ViT = {rocauc_2:.2f}')
    plt.legend(loc = 'lower right', fontsize=20)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.savefig(f"{i}_DensevsViT_100.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
    plt.show()
# %%
for i in pathologies:
    fpr1 = result_dic['result_100_w_ViT-pretrainedB16'][i]['fpr']
    tpr1 = result_dic['result_100_w_ViT-pretrainedB16'][i]['tpr']
    fpr2 = result_dic['result_100_resnet50_w'][i]['fpr']
    tpr2 = result_dic['result_100_resnet50_w'][i]['tpr']
    rocauc_1 = auc(fpr1, tpr1)
    rocauc_2 = auc(fpr2, tpr2)
    plt.figure(figsize=(10,10))
    plt.title(f'{i}', fontsize = 30)
    plt.plot(fpr1, tpr1, label = f'ROCAUC_ViT = {rocauc_1:.2f}')
    plt.plot(fpr2, tpr2, label = f'ROCAUC_Resnet50 = {rocauc_2:.2f}')
    plt.legend(loc = 'lower right', fontsize=30)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=30)
    plt.xlabel('False Positive Rate', fontsize=30)
    plt.savefig(f"{i}_resnet_vs_vit.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
    plt.show()
# %%
for i in pathologies:
    fpr1 = result_dic['result_100_ViT'][i]['fpr']
    tpr1 = result_dic['result_100_ViT'][i]['tpr']
    fpr2 = result_dic['result_100_dense_scratch'][i]['fpr']
    tpr2 = result_dic['result_100_dense_scratch'][i]['tpr']
    fpr3 = result_dic['result_100_resnet50_scratch'][i]['fpr']
    tpr3 = result_dic['result_100_resnet50_scratch'][i]['tpr']
    rocauc_1 = auc(fpr1, tpr1)
    rocauc_2 = auc(fpr2, tpr2)
    rocauc_3 = auc(fpr3, tpr3)
    plt.figure(figsize=(10,10))
    plt.title(f'{i}', fontsize = 30)
    plt.plot(fpr1, tpr1, label = f'ROCAUC_ViT = {rocauc_1:.2f}')
    plt.plot(fpr2, tpr2, label = f'ROCAUC_DenseNet = {rocauc_2:.2f}')
    plt.plot(fpr3, tpr3, label = f'ROCAUC_ResNet50 = {rocauc_2:.2f}')
    plt.legend(loc = 'lower right', fontsize=30)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=30)
    plt.xlabel('False Positive Rate', fontsize=30)
    plt.savefig(f"{i}_vit_vs_densevsresnet.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
    plt.show()
# %%

# %%
# ViT of different set
for i in pathologies:
    fpr1 = result_dic['result_100_ViT'][i]['fpr']
    tpr1 = result_dic['result_100_ViT'][i]['tpr']
    fpr2 = result_dic['result_60_ViT'][i]['fpr']
    tpr2 = result_dic['result_60_ViT'][i]['tpr']
    fpr3 = result_dic['result_30_ViT'][i]['fpr']
    tpr3 = result_dic['result_30_ViT'][i]['tpr']
    rocauc_1 = auc(fpr1, tpr1)
    rocauc_2 = auc(fpr2, tpr2)
    rocauc_3 = auc(fpr3, tpr3)
    plt.figure(figsize=(10,10))
    plt.title(f'{i}', fontsize = 30)
    plt.plot(fpr1, tpr1, label = f'ROCAUC_ViT_100% = {rocauc_1:.2f}')
    plt.plot(fpr2, tpr2, label = f'ROCAUC_ViT_67% = {rocauc_2:.2f}')
    plt.plot(fpr3, tpr3, label = f'ROCAUC_ViT_33% = {rocauc_3:.2f}')
    plt.legend(loc = 'lower right', fontsize=20)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=30)
    plt.xlabel('False Positive Rate', fontsize=30)
    plt.savefig(f"{i}_ViT.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
    plt.show()
# %%
result = pd.DataFrame()
for i in pathologies:
    y_true_1 = np.array(result_dic['result_100_ViT-pretrained'][i]['target'])
    y_pred_1 = 1*(np.array(result_dic['result_100_ViT-pretrained'][i]['pred'])>0.5)
    y_true_2 = np.array(result_dic['result_100_resnet50'][i]['target'])
    y_pred_2 = 1*(np.array(result_dic['result_100_resnet50'][i]['pred'])>0.5)
    
    f1_ViT = f1_score(y_true_1, y_pred_1)
    f1_resnet = f1_score(y_true_2, y_pred_2)
    df = pd.DataFrame({'Pathology' : [i,i], 'F1_score' : [f1_ViT, f1_resnet], 'Model' : ['ViT', 'ResNet50']})
    result = pd.concat([result, df], axis=0)

# barplot
sns.set(font_scale = 1.4)
plt.figure(figsize=(10,10))
plt.title('F1 score', fontsize = 20)
sns.barplot(data=result, x='Pathology', y= 'F1_score', hue='Model')
plt.savefig("F1_score_VIT_pre_vs_resnet_pre.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
# conf_mat = confusion_matrix(np.array(result_dic['result_100_ViT']['Consolidation']['target']),1*(np.array(result_dic['result_100_ViT']['Consolidation']['pred'])>0.5))
# %%
# compare models
def compare_models(model_list : list):
    for i in pathologies:

#%%
# Performance on different ViT models
result = pd.DataFrame()
for i in pathologies:
    y_true_1 = np.array(result_dic['result_100_ViT-pretrained'][i]['target'])
    y_pred_1 = 1*(np.array(result_dic['result_100_ViT-pretrained'][i]['pred'])>0.5)
    y_true_2 = np.array(result_dic['result_100_ViT-pretrainedB32'][i]['target'])
    y_pred_2 = 1*(np.array(result_dic['result_100_ViT-pretrainedB32'][i]['pred'])>0.5)
    y_true_3 = np.array(result_dic['result_100_ViT-pretrainedL'][i]['target'])
    y_pred_3 = 1*(np.array(result_dic['result_100_ViT-pretrainedL'][i]['pred'])>0.5)
    f1_ViTB16 = f1_score(y_true_1, y_pred_1)
    f1_ViTB32 = f1_score(y_true_2, y_pred_2)
    f1_ViTL32 = f1_score(y_true_3, y_pred_3)
    df = pd.DataFrame({'Pathology' : [i,i,i], 'F1_score' : [f1_ViTB16, f1_ViTB32, f1_ViTL32], 'Model' : ['ViTB16', 'ViTB32', 'ViTL32']})
    result = pd.concat([result, df], axis=0)

# barplot
sns.set(font_scale = 1.4)
plt.figure(figsize=(10,10))
plt.title('F1 score', fontsize = 20)
sns.barplot(data=result, x='Pathology', y= 'F1_score', hue='Model')
plt.savefig("F1_score_100_pretrained_VIT.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
#conf_mat = confusion_matrix(np.array(result_dic['result_100_ViT']['Consolidation']['target']),1*(np.array(result_dic['result_100_ViT']['Consolidation']['pred'])>0.5))
#%%
conf_mat1 = confusion_matrix(np.array(result_dic['result_100_ViT-B16_attn']['Atelectasis']['target']),1*(np.array(result_dic['result_100_ViT-B16_attn']['Atelectasis']['pred'])>0.5))
conf_mat2 = confusion_matrix(np.array(result_dic['result_100_ViT-B16_attn']['Cardiomegaly']['target']),1*(np.array(result_dic['result_100_ViT-B16_attn']['Cardiomegaly']['pred'])>0.5))
sns.set_palette('pastel')
sns.heatmap(conf_mat1, annot = True, cmap='YlGn', fmt='d')
plt.title('Atelectasis', fontsize=20)
plt.savefig("Confusion_matrix_Atelectasis.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
plt.show()
# %%
result = pd.DataFrame()
for i in pathologies:
    y_true_1 = np.array(result_dic['result_60_dense'][i]['target'])
    y_pred_1 = 1*(np.array(result_dic['result_60_dense'][i]['pred'])>0.5)
    y_true_2 = np.array(result_dic['result_60_ViT'][i]['target'])
    y_pred_2 = 1*(np.array(result_dic['result_60_ViT'][i]['pred'])>0.5)
    f1_dense = f1_score(y_true_1, y_pred_1)
    f1_ViT = f1_score(y_true_2, y_pred_2)
    df = pd.DataFrame({'Pathology' : [i,i], 'F1_score' : [f1_dense,f1_ViT], 'Model' : ['Densenet','ViT']})
    result = pd.concat([result, df], axis=0)

# barplot
plt.figure(figsize=(10,10))
plt.title('F1 score with (2/3) of data', fontsize = 20)
sns.barplot(data=result, x='Pathology', y= 'F1_score', hue='Model')
plt.savefig("F1_score_60.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
# %%
result = pd.DataFrame()
for i in pathologies:
    y_true_1 = np.array(result_dic['result_30_dense'][i]['target'])
    y_pred_1 = 1*(np.array(result_dic['result_30_dense'][i]['pred'])>0.5)
    y_true_2 = np.array(result_dic['result_30_ViT'][i]['target'])
    y_pred_2 = 1*(np.array(result_dic['result_30_ViT'][i]['pred'])>0.5)
    f1_dense = f1_score(y_true_1, y_pred_1)
    f1_ViT = f1_score(y_true_2, y_pred_2)
    df = pd.DataFrame({'Pathology' : [i,i], 'F1_score' : [f1_dense,f1_ViT], 'Model' : ['Densenet','ViT']})
    result = pd.concat([result, df], axis=0)

# barplot
plt.figure(figsize=(10,10))
plt.title('F1 score with (1/3) of data', fontsize = 20)
sns.barplot(data=result, x='Pathology', y= 'F1_score', hue='Model')
plt.savefig("F1_score_30.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
# %%
# %%
result = pd.DataFrame()
for i in pathologies:
    y_true_1 = np.array(result_dic['result_100_dense'][i]['target'])
    y_pred_1 = 1*(np.array(result_dic['result_100_dense'][i]['pred'])>0.5)
    y_true_2 = np.array(result_dic['result_100_ViT'][i]['target'])
    y_pred_2 = 1*(np.array(result_dic['result_100_ViT'][i]['pred'])>0.5)
    cohen_dense = cohen_kappa_score(y_true_1, y_pred_1)
    cohen_ViT = cohen_kappa_score(y_true_2, y_pred_2)
    df = pd.DataFrame({'Pathology' : [i,i], 'Cohen kappa score' : [cohen_dense,cohen_ViT], 'Model' : ['Densenet','ViT']})
    result = pd.concat([result, df], axis=0)

# barplot
sns.set(font_scale = 1.4)
plt.figure(figsize=(10,10))
plt.title('Cohen kappa score', fontsize = 20)
sns.barplot(data=result, x='Pathology', y= 'Cohen kappa score', hue='Model')
plt.savefig("Cohen_kappa_score_100.png", edgecolor = "white", facecolor='white', transparent=False, frameon = True)
# %%
import torchvision
import torch
from utils.ChexpertDataset import CheXpertDataSet
import torchvision.transforms as transforms
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    C, H, W = x.shape
    x = x.reshape(C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(1, 3, 0, 2, 4) # [B, H', W', C, p_H, p_W]
    x = x.flatten(0,1)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

pathFileTrain = Path(__file__).parents[2].joinpath("./CheXpert-v1.0-small/train_30.csv")
train_set = CheXpertDataSet(pathFileTrain)

NUM_IMAGES = 1
Chexpert_image = train_set[0][0]



plt.figure(figsize=(20,20))
#plt.title("Image examples of the Chexpert dataset")
plt.imshow(Chexpert_image)
plt.axis('off')
plt.savefig('Original.png')
plt.show()
plt.close()
# %%
convert_tensor = transforms.Compose([transforms.ToTensor(), transforms.RandomResizedCrop(256)])
img_tensor = convert_tensor(Chexpert_image)

# %%
img_patches = img_to_patch(img_tensor, patch_size=32, flatten_channels=False)
plt.figure(figsize=(20,20))
plt.subplot(111)
#plt.title("Image grid")
img_grid = torchvision.utils.make_grid(img_patches, nrow=8, normalize=True, pad_value=0.9)
img_grid = img_grid.permute(1, 2, 0)
plt.imshow(img_grid)
plt.axis('off')
plt.savefig('Grid.png')
plt.show()
plt.close()
# %%
plt.figure(figsize=(20,20))
plt.subplot(111)
#plt.title("Images as input sequences of patches")
img_grid = torchvision.utils.make_grid(img_patches, nrow=1, normalize=True, pad_value=0.9)
img_grid = img_grid.permute(1, 2, 0)
plt.imshow(img_grid)
plt.axis('off')
plt.savefig('Patches.png')
plt.show()
plt.close()
# %%
sns.set(rc = {'figure.figsize':(60,40)})
#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('pastel')
models = ['ViT', 'ResNet50', 'DeiT', 'CrossViT', 'DeepViT']
inference_time = [3.21, 13.94, 7.07, 12.022, 2.377]
params  = [85.8, 23.5, 62.3, 42.2, 101]
sns.set_style("darkgrid")
sns.scatterplot(x = params, y = inference_time, style = models, s=100)
plt.xlabel('Model parameters (in M)', fontsize=15)
plt.ylabel('Inference (images / s)', fontsize = 15)
plt.title('Comparison of Inference time', fontsize = 15)
plt.savefig('inference_time.png',edgecolor = "white", facecolor='white', transparent=False, frameon = True)
# %%
