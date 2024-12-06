

import os.path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torchvision.models import ResNet50_Weights
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# define the folder to save the tpl files
features_dir = 'F:/原电脑深度学习相关/多模态谣言检测数据/微博/含噪微博原图像特征'

# define the folder where the images are stored
data_dir = 'F:/原电脑深度学习相关/多模态谣言检测数据/微博/ALL_pic_noise'
# Create the feature tensor directory if it doesn't exist
if not os.path.exists(features_dir):
    os.makedirs(features_dir)
'''
# Define the image transformations to apply
transform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])

# Define the ResNet50 model and change the last layer to output 1024 features
resnet50_feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50_feature_extractor.fc = nn.Linear(2048, 1024)

# Freeze the weights of all layers in the model except the last one
for param in resnet50_feature_extractor.parameters():
    param.requires_grad = False

# Loop through all the images in the data directory
for filename in os.listdir(data_dir):
    # Load the image and apply the transformations
    img = Image.open(os.path.join(data_dir, filename)).convert("RGB")
    img1 = transform1(img)

    # Extract the features from the image using the ResNet50 model
    x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
    y = resnet50_feature_extractor(x)
    y = y.detach().squeeze().numpy()

    # Save the features tensor as a tpl file with the same name as the image
    features_path = os.path.join(features_dir, os.path.splitext(filename)[0] + '.pt')
    torch.save(torch.tensor(y), features_path)

    # Print the shape of the saved features tensor
    print(f"Saved tensor with shape {y.shape} to {features_path}")
    print(y)



("--------------------------------------------------------------------------")
import os
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet50_Weights
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the directory to save the feature tensors
features_dir = 'C:/Users/.../Resnet50_feature_twitter_ELA_image_tensor'

# Define the directory where the images are stored
data_dir = 'C:/Users/.../Twitter数据集/ELA_Allpic'

# Create the feature tensor directory if it doesn't exist
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Set the model to evaluation mode
resnet50.eval()

# Loop through all the images in the data directory
for filename in os.listdir(data_dir):
    # Load the image and apply the transformations
    img = Image.open(os.path.join(data_dir, filename)).convert("RGB")
    img_transformed = transform(img)

    # Add an extra batch dimension since PyTorch treats all inputs as batches
    img_batch = img_transformed.unsqueeze(0)

    # Extract features using the ResNet50 model
    with torch.no_grad():  # Disable gradient calculation for efficiency
        outputs = resnet50(img_batch)

    # Save the extracted features as a .pt file
    features_path = os.path.join(features_dir, os.path.splitext(filename)[0] + '.pt')
    torch.save(outputs, features_path)

    # Print the path where the features are saved
    print(f"Saved features to {features_path}")
'''

'''
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet50_Weights
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 定义保存特征文件的文件夹路径
features_dir = 'F:/原电脑深度学习相关/多模态谣言检测数据/推特/含噪的推特ELA图像特征'

# 定义存储图像的文件夹路径
data_dir = 'F:/原电脑深度学习相关/多模态谣言检测数据/推特/ELA_twitter_allpic_noise'

# 如果特征张量目录不存在，则创建该目录
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

# 定义图像转换
transform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])

# 定义ResNet50模型，并更改最后一层，以输出1024个特征
resnet50_feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50_feature_extractor.fc = nn.Linear(2048, 1024)

# 冻结所有层的权重，除了最后一层
for param in resnet50_feature_extractor.parameters():
    param.requires_grad = False

# 将模型设置为评估模式
resnet50_feature_extractor.eval()

# 遍历数据目录中的所有图像
for filename in os.listdir(data_dir):
    # 尝试加载和处理图像，然后提取特征
    try:
        # 加载图像并应用预定义的转换
        img = Image.open(os.path.join(data_dir, filename)).convert("RGB")
        img_transformed = transform1(img)

        # 不计算梯度，以节省内存和计算资源
        with torch.no_grad():
            # 使用ResNet50模型从图像中提取特征
            features = resnet50_feature_extractor(torch.unsqueeze(img_transformed, dim=0))
            features = features.squeeze().numpy()

        # 以与图像同名的.pt文件保存特征张量
        features_path = os.path.join(features_dir, os.path.splitext(filename)[0] + '.pt')
        torch.save(features, features_path)

        # 打印保存的特征张量的形状
        print(f"Saved tensor with shape {features.shape} to {features_path}")
    except Exception as e:
        # 打印出现的任何错误消息
        print(f"Error processing {filename}: {e}")
'''

features_dir = 'F:/原电脑深度学习相关/多模态谣言检测数据/推特/含噪的推特ELA图像特征'

# define the folder where the images are stored
data_dir = 'F:/原电脑深度学习相关/多模态谣言检测数据/推特/ELA_twitter_allpic_noise'
# Create the feature tensor directory if it doesn't exist
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

# Define the image transformations to apply
transform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])

# Define the ResNet50 model and change the last layer to output 1024 features
resnet50_feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50_feature_extractor.fc = nn.Linear(2048, 1024)

# Freeze the weights of all layers in the model except the last one
for param in resnet50_feature_extractor.parameters():
    param.requires_grad = False

# Loop through all the images in the data directory
for filename in os.listdir(data_dir):
    # Load the image and apply the transformations
    img = Image.open(os.path.join(data_dir, filename)).convert("RGB")
    img1 = transform1(img)

    # Extract the features from the image using the ResNet50 model
    x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
    y = resnet50_feature_extractor(x)
    y = y.detach().squeeze()#.numpy()

    # Save the features tensor as a tpl file with the same name as the image
    features_path = os.path.join(features_dir, os.path.splitext(filename)[0] + '.pt')
    torch.save(torch.tensor(y), features_path)

    # Print the shape of the saved features tensor
    print(f"Saved tensor with shape {y.shape} to {features_path}")
    #print(y)