from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model.ResNet50 import Resnet50
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = r'C:/Users/prott/OneDrive/Desktop/Thesis/cheXpert/chexpert-lightning-prottay/lightning_logs/version_62/checkpoints/last.ckpt'
model = Resnet50.load_from_checkpoint(PATH)

# print(model.learning_rate)
# model.eval()

img_path = 'view1_frontal.jpg'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform2 = transforms.Compose([
    transforms.Resize((224, 224))
])

#y_true, y_pred = model(CheXpertData)
im = Image.open(img_path).convert('RGB')
x = transform(im)
target_layers = [model.resnet50.layer4[-1]]
input_tensor = x.unsqueeze(0)# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model.resnet50, target_layers=target_layers, use_cuda=True)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.
targets = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor)

# In this example grayscale_cam has only one image in the batch:#
grayscale_cam = grayscale_cam[0, :]
x = np.array(transform2(im))/255
#breakpoint()
visualization = show_cam_on_image(x, grayscale_cam, use_rgb=True)

# imgOriginal = cv2.imread(img_path, 1)
# imgOriginal = cv2.resize(imgOriginal, (224, 224))


# heatmap = cv2.applyColorMap(visualization, cv2.COLORMAP_JET)

# img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)            
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #plt.title(label)
# plt.imshow(img)
# plt.plot()
# plt.axis('off')
# #plt.savefig(pathOutputFile)
# plt.show()

im = Image.fromarray(visualization)
im.show()
im.save('heatmap_resnet.png')