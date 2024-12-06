import argparse
import itertools
import time

import numpy as np
import pandas as pd
import torch
# 谣言为1，非谣言为0
# BERT文本特征为BERT_text_feature,GCN文本特征为GCN_text_feature,Resnet50特征为Resnet50_image,ELA特征为ELA_image
# 每个特征均为一维
import os
import random

from torch import nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.testing._internal.codegen.random_topo_test import DEVICE
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import get_linear_schedule_with_warmup

from MCAN import DECM



                                        
("------------------------------------------------------------------------------------")
# start a new wandb run to track this script
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# wandb.init(project="Rumor detection base data enhance", entity="zhangyaozeng", name="twitter830")

"---------------------------------------------------------------------------------------------"

single_model_DIM = 768
Text_DIM = 768
style_DIM = 200
Image_DIM = 1024
ELA_DIM = 1024

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, choices=["DECM", "text_only", "GCN_only", "image_only", "ELA_only"], default="DECM",
)

parser.add_argument("--dataset", type=str)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_seq_length", type=int, default=85)
# parser.add_argument("--n_layers", type=int, default=4)
# parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--cross_n_layers", type=int, default=4)
parser.add_argument("--cross_n_heads", type=int, default=4)
parser.add_argument("--fusion_dim", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--epochs", type=int, default=50)

parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)

parser.add_argument("--learning_rate", type=float, default=0.0001)  # 0.000005
parser.add_argument("--learning_rate_a", type=float, default=0.003)
parser.add_argument("--learning_rate_h", type=float, default=0.0003)
parser.add_argument("--learning_rate_v", type=float, default=0.003)
parser.add_argument("--warmup_ratio", type=float, default=0.07178)
parser.add_argument("--save_weight", type=str, choices=["True", "False"], default="False")
args = parser.parse_args()


# Define the path to the folder containing the txt files


def return_unk():
    return 0


"---------------------------------------------------------------------------------------------"
"文件位置"
folder_way = "F:\\原电脑深度学习相关\\谣言信息检测-博一工程\\twitter(14434条)所有特征\\new_twitter_list"


# 定义输入特征的名称
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, data_id, text_features, GCN_features, image_features, ELA_features, data_label):
        self.data_id = data_id
        self.text_features = text_features
        self.GCN_features = GCN_features
        self.image_features = image_features
        self.ELA_features = ELA_features
        self.data_label = data_label


# Initialize an empty dictionary to store the data

# 获取数据的字典
# Loop through each file in the folder
def data_dict(folder_path):
    data_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                # Read the lines from the file
                lines = f.readlines()
                # Extract the data from the lines
                data_id = lines[0].strip()
                text_content = lines[1].strip()
                image_id = lines[2].strip()
                data_label = lines[3].strip()
                # Add the data to the dictionary
                data_dict[data_id] = {"text_content": text_content, "image_id": image_id, "data_label": data_label}

    return data_dict


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# 特征拼接并获取特征长度
def concatefeature(BERT_text_feature, GCN_text_feature, Resnet50_image, ELA_image):
    Text_GCN_embadding = torch.cat([BERT_text_feature, GCN_text_feature], dim=1)
    Image_ELA_embadding = torch.cat([Resnet50_image, ELA_image], dim=1)
    Text_GCN_embadding_len = Text_GCN_embadding.size(1)
    Image_ELA_embadding_len = Text_GCN_embadding.size(1)
    return Text_GCN_embadding, Image_ELA_embadding, Text_GCN_embadding_len, Image_ELA_embadding_len


# Assuming text features are stored in "text_features" folder and image features are stored in "image_features" folder
text_fea = "text_features"
GCN_fea = "GCN_features"
image_fea = "image_features"
ELA_fea = "ELA_features"

folder_path = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/new_twitter_list"
text_features_wa = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/twitter_bert_sentence_embedding"
GCN_features_wa = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/GCN_twittertext_feature"
image_features_wa = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/twitter_pic_Resnet50"
ELA_features_wa = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/twitter_pic_ELA_Resnet50"


# Define a function to load features from file
def load_features_from_file(filename):
    # Load feature tensor from file
    feature_tensor = torch.load(filename)
    # Convert feature tensor to a list and return
    feature_tensor = feature_tensor.view(1, -1)
    return feature_tensor.tolist()[0]


# global data_item

def get_feature(data_item,
                text_features_folder,
                GCN_features_folder,
                image_features_folder,
                ELA_features_folder):
    # Initialize a list to store all the InputFeatures
    features = []
    # data_item = data_dict()
    # Loop through each data_id in the data_dict
    for data_id, data_info in data_item.items():
        # data_id
        data_id = data_id
        # Load text content
        text_content = data_info["text_content"]
        # Load image id
        image_id = data_info["image_id"]

        # Load text features
        text_features_filename = f"{text_features_folder}/{data_id}.pt"
        text_features = load_features_from_file(text_features_filename)
        # Load GCN features
        GCN_features_filename = f"{GCN_features_folder}/{data_id}.pt"
        GCN_features = load_features_from_file(GCN_features_filename)
        # Load image features
        image_features_filename = f"{image_features_folder}/{data_id}.pt"
        image_features = load_features_from_file(image_features_filename)
        # Load ELA features
        ELA_features_filename = f"{ELA_features_folder}/{data_id}.pt"
        ELA_features = load_features_from_file(ELA_features_filename)
        # Load data label
        data_label = data_info["data_label"]
        # data_id = torch.tensor(list(data_id), dtype=torch.int32)
        # data_id = torch.squeeze(data_id)
        # Construct InputFeatures object
        features.append(
            InputFeatures(
                data_id=data_id,
                text_features=text_features,
                GCN_features=GCN_features,
                image_features=image_features,
                ELA_features=ELA_features,
                data_label=data_label,
            )
        )
        # Add InputFeatures to the list
    return features  # returns a list of features


# get_feature_data_item指get_feature当中的features
# 获取数据集（根据features列表  data_dict
def get_appropriate_dataset(get_feature_data_item):
    features = get_feature_data_item
    all_data_id = torch.tensor([int(f.data_id) for f in features], dtype=torch.long)

    # all_data_id = torch.tensor([f.data_id for f in features], dtype=torch.long)
    all_text = torch.tensor([f.text_features for f in features], dtype=torch.float)

    all_GCN = torch.tensor([f.GCN_features for f in features], dtype=torch.float)

    all_image = torch.tensor([f.image_features for f in features], dtype=torch.float)

    all_ELA = torch.tensor([f.ELA_features for f in features], dtype=torch.float)

    all_label_ids = torch.tensor([int(f.data_label) for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_data_id,
        all_text,
        all_GCN,
        all_image,
        all_ELA,
        all_label_ids,
    )
    return dataset


def set_up_data_loader(dataset_):
    # Load the train-test information from tran_test.txt
    with open("F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter(14434条)所有特征/twitter_0608.txt",
              "r") as file:
        lines = file.readlines()

    # Parse the train-test information
    train_indices = []
    test_indices = []
    for line in lines:
        values = line.strip().split("\t")
        if len(values) != 3:
            continue  # Skip the line if it does not contain the expected number of values
        data_id, train_label, target_label = values
        data_id = int(data_id)
        print(data_id)
        if train_label == "train":
            train_indices.append(data_id)
        elif train_label == "test":
            test_indices.append(data_id)

    # Create the final datasets and dataloaders using the indices
    train_dataset = Subset(dataset_, train_indices)
    test_dataset = Subset(dataset_, test_indices)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    dev_dataloader = test_dataloader
    return train_dataloader, dev_dataloader, test_dataloader


def set_up_data_loader(dataset_):
    # Load the train-test information from tran_test.txt
    with open("F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter(14434条)所有特征/twitter_0608.txt", "r") as file:
        lines = file.readlines()

    # Extract the data IDs and labels from the text file
    data_ids = []
    train_test_labels = []
    target_labels = []
    for line in lines:
        data_id, train_test_label, target_label = line.strip().split('\t')
        data_ids.append(int(data_id))
        train_test_labels.append(train_test_label)
        target_labels.append(int(target_label))

    # Create indices for train and test samples
    train_indices = [i for i, label in enumerate(train_test_labels) if label == 'train']
    valid_indices = [i for i, label in enumerate(train_test_labels) if label == 'test']

    test_indices = [i for i, label in enumerate(train_test_labels) if label == 'test']

    # Create subsets based on the indices
    train_subset = torch.utils.data.Subset(dataset_, train_indices)
    valid_subset = torch.utils.data.Subset(dataset_, valid_indices)
    test_subset = torch.utils.data.Subset(dataset_, test_indices)

    # Create data loaders for training and testing sets
    train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(valid_subset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, eval_dataloader, test_dataloader


# 打印检查dataloeader内容
# Printing the data in the training set DataLoader
def print_dataloader(dataloader_):
    print("Set:")
    for batch in dataloader_:
        # Unpack the batch
        input_ids = batch[0]
        text = batch[1]
        GCN = batch[2]
        image = batch[3]
        ELA = batch[4]
        labels = batch[5]

    # Print the batch data
    return print("Input IDs:", input_ids), print("text:", text), print("GCN:", GCN), print("image:", image), print(
        "ELA:", ELA), print("Labels:", labels)


# 训练集epoch
def train_epoch(model, train_dataloader, optimizer, scheduler, loss_fct):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

        batch = tuple(t.to(DEVICE) for t in batch)
        (
            data_id,
            text_features,
            GCN_features,
            image_features,
            ELA_features,
            data_label,
        ) = batch
        # print(batch)
        outputs = model(text_features, GCN_features, image_features, ELA_features)
        # outputs = model(text_features, GCN_features)
        # print("train")
        logits = outputs[0]
        print(data_label)
        print(logits.view(-1))
        print(data_label.view(-1))
        # 保存测试集结果

        # print(logits)
        loss = loss_fct(logits.view(-1), data_label.view(-1))

        tr_loss += loss.item()
        nb_tr_examples += data_id.size(0)
        nb_tr_steps += 1

        loss.backward()

        for o_i in range(len(optimizer)):
            optimizer[o_i].step()
            scheduler[o_i].step()

        model.zero_grad()

    return tr_loss / nb_tr_steps


# 验证集epoch
def eval_epoch(model, dev_dataloader, loss_fct):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            (
                data_id,
                text_features,
                GCN_features,
                image_features,
                ELA_features,
                data_label,
            ) = batch
            # print(batch)
            outputs = model(text_features, GCN_features, image_features, ELA_features)
            # outputs = model(text_features, GCN_features)
            # print("eval")
            logits = outputs[0]

            print(data_label)

            loss = loss_fct(logits.view(-1), data_label.view(-1))

            dev_loss += loss.item()
            nb_dev_examples += data_id.size(0)
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


# 测试集epoch
def test_epoch(model, test_data_loader, loss_fct):
    """ Epoch operation in evaluation phase """
    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    all_labels = []
    feature_vectors = []  # 用于保存每次测试的 fused_hidden
    labels = []  # List to store labels
    inference_times = []  # List to store inference times
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data_loader, desc="Iteration")):

            start_time = time.time()  # Record start time for inference

            batch = tuple(t.to(DEVICE) for t in batch)
            (
                data_id,
                text_features,
                GCN_features,
                image_features,
                ELA_features,
                data_label,
            ) = batch
            # print(batch)
            outputs = model(text_features, GCN_features, image_features, ELA_features)
            # outputs = model(text_features, GCN_features)
            fused_hidden = outputs[1]  # Assuming `outputs[1]` contains `fused_hidden`

            fusion_weights = model.fc[0].weight.detach().cpu().numpy()
            feature_vector = np.dot(fused_hidden.detach().cpu().numpy(), fusion_weights.T)
            feature_vectors.append(feature_vector.T)
            # 在所有特征向量的第二个维度上进行叠加

            labels.append(data_label.detach().cpu().numpy())
            # print("test")
            logits = outputs[0]
            print(logits.view(-1))
            print(data_label.view(-1))

            tmp_eval_loss = loss_fct(logits.view(-1), data_label.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            logits = torch.sigmoid(logits)

            if len(preds) == 0:
                preds = logits.detach().cpu().numpy()
                all_labels = data_label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_labels = np.append(
                    all_labels, data_label.detach().cpu().numpy(), axis=0
                )

            end_time = time.time()  # Record end time for inference
            inference_time = end_time - start_time
            inference_times.append(inference_time)
        print("推理时间：", inference_times)
        eval_loss = eval_loss / nb_eval_steps
        preds = np.squeeze(preds)
        print(preds)
        all_labels = np.squeeze(all_labels)

    '''
    # 在特征向量的第二个维度上进行叠加
    complete_feature_vectors = np.concatenate(feature_vectors, axis=1)

    np.save("F:\\原电脑深度学习相关\\谣言检测深度学习工程MTAP\\模型参数\\SNE\\weibo_fusion_dim.npy",
            np.array(complete_feature_vectors))
    with open("F:\\原电脑深度学习相关\\谣言检测深度学习工程MTAP\\模型参数\\SNE\\weibo_image_labels.txt", "w") as file:
        for label in labels:
            file.write(str(label) + "\n")
    '''

    return preds, all_labels, eval_loss


# 测试集分数
def test_score_model(model, test_data_loader, loss_fct, exclude_zero=False):
    predictions, y_test, test_loss = test_epoch(model, test_data_loader, loss_fct)

    predictions = predictions.round()

    f_score = f1_score(y_test, predictions, average="weighted")
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy, "F score:", f_score)
    return accuracy, f_score, test_loss


# 测试集分数
def test_score_model(model, test_data_loader, loss_fct, exclude_zero=False):
    predictions, y_test, test_loss = test_epoch(model, test_data_loader, loss_fct)

    predictions = predictions.round()
    print(predictions)
    f_score = f1_score(y_test, predictions, average=None)
    accuracy = accuracy_score(y_test, predictions)
    precision_label = precision_score(y_test, predictions, average=None)
    recall_ = recall_score(y_test, predictions, average=None)

    print("Accuracy:", accuracy, "F score:", f_score, "recall score", recall_, "predictions_label", precision_label)
    return accuracy, f_score, test_loss, recall_, precision_label


# 训练
def train(
        model,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        loss_fct,
        patience,
):
    best_valid_loss = 9e+9
    run_name = str(wandb.run.id)
    valid_losses = []
    patience_counter = 0  # 初始化早停计数器
    n_epochs = args.epochs
    total_train_time = 0  # 用于记录总训练时间
    for epoch_i in range(n_epochs):
        start_time = time.time()  # 记录每个 epoch 的开始时间

        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, loss_fct
        )

        valid_loss = eval_epoch(model, dev_dataloader, loss_fct)

        valid_losses.append(valid_loss)
        print(
            "\nepoch:{},train_loss:{}, valid_loss:{}".format(
                epoch_i, train_loss, valid_loss
            )
        )
        # accuracy, f_score, test_loss, recall_, precision_label
        test_accuracy, test_f_score, test_loss, recall_, precision_label = test_score_model(
            model, test_dataloader, loss_fct
        )

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_test_accuracy = test_accuracy
            best_valid_test_fscore = test_f_score
            best_valid_test_recall_ = recall_
            best_valid_precision_label = precision_label

            f1 = best_valid_test_fscore[0]
            r1 = best_valid_test_recall_[0]
            p1 = best_valid_precision_label[0]

            f2 = best_valid_test_fscore[1]
            r2 = best_valid_test_recall_[1]
            p2 = best_valid_precision_label[1]

        if args.save_weight == "True":
            torch.save(model.state_dict(), './best_weights/' + run_name + '.pt')

        # we report test_accuracy of the best valid loss (best_valid_test_accuracy)
        wandb.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "test_loss": test_loss,
                "best_valid_loss": best_valid_loss,
                "best_valid_test_accuracy": best_valid_test_accuracy,

                "0_best_valid_test_fscore": f1,
                "0_best_valid_test_recall_": r1,
                "0_best_valid_precision_label": p1,

                "1_best_valid_test_fscore": f2,
                "1_best_valid_test_recall_": r2,
                "1_best_valid_precision_label": p2
            }
        )

        end_time = time.time()  # 记录每个 epoch 的结束时间
        epoch_time = end_time - start_time  # 计算每个 epoch 的训练时间
        total_train_time += epoch_time  # 累加到总训练时间中
        print(f"Epoch {epoch_i} took {epoch_time} seconds")

        # 检查是否需要早停
        if patience_counter >= patience:
            print(f"Early stopping! Validation loss didn't improve for {patience} epochs.")
            break
        elif valid_loss > best_valid_loss:
            patience_counter += 1
        else:
            patience_counter = 0
    print(f"Total training time: {total_train_time} seconds")


def get_optimizer_scheduler(params, num_training_steps, learning_rate=1e-5):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


("计算模型的参数和存储开销")


def calculate_model_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def calculate_model_storage(model):
    #    dummy_input = [torch.randn(1, *input_shape) for input_shape in model.input_shapes]
    model_size = torch.cuda.memory_allocated() if torch.cuda.is_available() else torch.tensor(0)
    return model_size


("计算模型的参数和存储开销")


def prep_for_training(num_training_steps, get_feature_data_item):
    dataset = get_appropriate_dataset(get_feature_data_item)
    text_model = dataset.tensors[1]
    GCN_model = dataset.tensors[2]
    image_model = dataset.tensors[3]
    ELA_model = dataset.tensors[4]

    model = DECM(text_model, GCN_model, image_model, ELA_model, args)
    # model = DECM(text_model, GCN_model, args)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    loss_fct = BCEWithLogitsLoss()
    loss_fct = FocalLoss()
    # Prepare optimizer
    # used different learning rates for different componenets.

    params = list(model.named_parameters())

    optimizer_l, scheduler_l = get_optimizer_scheduler(
        params, num_training_steps, learning_rate=args.learning_rate
    )

    optimizers = [optimizer_l]
    schedulers = [scheduler_l]
    return model, optimizers, schedulers, loss_fct


def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


# Define the hyperparameter grid
hyperparameter_grid = {
    'learning_rate': [0.001, 0.0001, 0.00001],
    'fusion_dim': [64, 128, 256],  # ,32
    'dropout': [0.1, 0.2, 0.5],  # ,32
    'cross_n_layers': [1, 2, 3, 4, 5, 6],  # 1, 2, 4, 5,
    'cross_n_heads': [1, 2, 4, 8],
    # Add more hyperparameters as needed
}

# Define the hyperparameter grid
hyperparameter_grid = {
    'learning_rate': [0.001],
    'fusion_dim': [256],  # ,32
    'dropout': [0.2],  # ,32
    'cross_n_layers': [3],  # 1, 2, 4, 5,
    'cross_n_heads': [4],
    # Add more hyperparameters as needed
}

def main():
    for hyperparams in itertools.product(*hyperparameter_grid.values()):
        hyperparameter_dict = dict(zip(hyperparameter_grid.keys(), hyperparams))

        # 初始化一个新的 WandB 运行
        run_name = f"t_run_lr_{hyperparameter_dict['learning_rate']}_hs_{hyperparameter_dict['fusion_dim']}_nl_{hyperparameter_dict['cross_n_layers']}_nh_{hyperparameter_dict['cross_n_heads']}_dp_{hyperparameter_dict['dropout']}"
        print(run_name)
        wandb.init(project="MTAP_twitter_zt_08", entity="zhangyaozeng", name=run_name)
        wandb.config.update(args)

        args.fusion_dim = hyperparameter_dict['fusion_dim']
        args.cross_n_layers = hyperparameter_dict['cross_n_layers']
        args.learning_rate = hyperparameter_dict['learning_rate']
        args.cross_n_heads = hyperparameter_dict['cross_n_heads']
        args.dropout = hyperparameter_dict['dropout']

        if args.seed == -1:
            seed = random.randint(0, 9999)
            print("seed", seed)
        else:
            seed = args.seed

        wandb.config.update({"seed": seed}, allow_val_change=True)

        set_random_seed(seed)

        folder_path = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitte所有特征/new_twitter_list"
        text_features_wa = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/twitter_bert_sentence_embedding"
        GCN_features_wa = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/GCN_twittertext_feature"

        image_features_wa = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/twitter_pic_Resnet50"
        ELA_features_wa = "F:/原电脑深度学习相关/谣言信息检测-博一工程/twitter所有特征/twitter所有特征/twitter_pic_ELA_Resnet50"
        data_ite = data_dict(folder_path)

        get_feature_data_item = get_feature(data_ite,
                                            text_features_wa,
                                            GCN_features_wa,
                                            image_features_wa,
                                            ELA_features_wa)

        dataset_ = get_appropriate_dataset(get_feature_data_item)
        # print(dataset_)
        # train_dataloader, dev_dataloader, test_dataloader = set_up_data_loader()
        train_dataloader, dev_dataloader, test_dataloader = set_up_data_loader(dataset_)
        # print_dataloader(train_dataloader)
        # print_dataloader(dev_dataloader)
        # print_dataloader(test_dataloader)
        print(len(train_dataloader), len(dev_dataloader), len(test_dataloader))
        print("Dataset Loaded")

        num_training_steps = len(train_dataloader) * args.epochs

        model, optimizers, schedulers, loss_fct = prep_for_training(
            num_training_steps, get_feature_data_item
        )

        # 计算模型的参数量和存储开销
        num_params = calculate_model_parameters(model)
        model_storage = calculate_model_storage(model)

        print("Model Loaded: ", args.model)
        train(
            model,
            train_dataloader,
            dev_dataloader,
            test_dataloader,
            optimizers,
            schedulers,
            loss_fct,
            patience=args.patience,  # 设置早停容忍度,  # 设置早停容忍度,
        )

        print(f"Number of parameters: {num_params}")
        print(f"Model storage: {model_storage} bytes")

        wandb.finish()


if __name__ == "__main__":
    main()

'''
# 已知DECM模型输入为（Text_embadding, Image_embadding, Text_embadding_len, Image_embadding_len）

'''
