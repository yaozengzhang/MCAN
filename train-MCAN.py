import argparse

import numpy as np
import torch
# 谣言为1，非谣言为0
# BERT文本特征为BERT_text_feature,GCN文本特征为GCN_text_feature,Resnet50特征为Resnet50_image,ELA特征为ELA_image
# 每个特征均为一维
import os
import random

import wandb
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.testing._internal.codegen.random_topo_test import DEVICE
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import get_linear_schedule_with_warmup
# from DECMsinglemodal import DECM
from DECMmodal import DECM
# from DECMonefeature import DECM
import torch

# start a new wandb run to track this script
wandb.init(project="Rumor detection base data enhance", entity="zhangyaozeng", name="test")

single_model_DIM = 768
Text_DIM = 768
style_DIM = 200
Image_DIM = 1024
ELA_DIM = 1024


def return_unk():
    return 0


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, choices=["DECM", "text_only", "GCN_only", "image_only", "ELA_only"], default="DECM",
)

parser.add_argument("--dataset", type=str)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=85)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--cross_n_layers", type=int, default=1)
parser.add_argument("--cross_n_heads", type=int, default=4)
parser.add_argument("--fusion_dim", type=int, default=172)
parser.add_argument("--dropout", type=float, default=0.2366)
parser.add_argument("--epochs", type=int, default=50)

parser.add_argument("--seed", type=int, default=100)

parser.add_argument("--learning_rate", type=float, default=0.000005)  # 0.000005
parser.add_argument("--learning_rate_a", type=float, default=0.003)
parser.add_argument("--learning_rate_h", type=float, default=0.0003)
parser.add_argument("--learning_rate_v", type=float, default=0.003)
parser.add_argument("--warmup_ratio", type=float, default=0.07178)
parser.add_argument("--save_weight", type=str, choices=["True", "False"], default="False")

args = parser.parse_args()
# Define the path to the folder containing the txt files

folder_way = "E:\\谣言检测深度学习工程\\twitter数据集（10620条）所有特征\\twitter_list"


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
text_features_way = "E:\\谣言检测深度学习工程\\twitter数据集（10620条）所有特征\\twitter_bert_sentence_embedding",
GCN_features_way = "E:\\谣言检测深度学习工程\\twitter数据集（10620条）所有特征\\GCN_twittertext_feature_tensor",
image_features_way = "E:\\谣言检测深度学习工程\\twitter数据集（10620条）所有特征\\Resnet50_images_tensor",
ELA_features_way = "E:\\谣言检测深度学习工程\\twitter数据集（10620条）所有特征\\Resnet50_ELA_image_tensor"


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
    #    all_label_ids = torch.tensor([f.data_label for f in features], dtype=torch.float)
    #    all_label_ids = torch.tensor([f.data_label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_data_id,
        all_text,
        all_GCN,
        all_image,
        all_ELA,
        all_label_ids,
    )
    return dataset


'''

# 划分数据集
def set_up_data_loader(dataset_, train_ratio=0.7, dev_ratio=0.1, test_ratio=0.2):
    # Calculate the sizes of each split based on the ratios
    dataset = dataset_
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_dev = int(dev_ratio * num_samples)
    num_test = num_samples - num_train - num_dev

    train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_dev, num_test]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    return train_dataloader, dev_dataloader, test_dataloader
'''


def set_up_data_loader(dataset_, train_ratio=0.7, dev_ratio=0.1, test_ratio=0.2):
    # Get all the indices corresponding to the label 0 and 1
    label_0_indices = [i for i, label in enumerate(dataset_.tensors[5]) if label == 0]
    label_1_indices = [i for i, label in enumerate(dataset_.tensors[5]) if label == 1]

    # Calculate the sizes of each split based on the ratios and the size of each label group
    num_samples_0 = len(label_0_indices)
    num_samples_1 = len(label_1_indices)
    print(num_samples_0)
    print(num_samples_1)
    num_train_0 = int(train_ratio * num_samples_0)
    num_dev_0 = int(dev_ratio * num_samples_0)
    num_test_0 = num_samples_0 - num_train_0 - num_dev_0

    num_train_1 = int(train_ratio * num_samples_1)
    num_dev_1 = int(dev_ratio * num_samples_1)
    num_test_1 = num_samples_1 - num_train_1 - num_dev_1

    # Calculate the sizes of the final splits
    num_train = num_train_0 + num_train_1
    num_dev = num_dev_0 + num_dev_1
    num_test = num_test_0 + num_test_1

    # Randomly shuffle the indices corresponding to each label group
    random.shuffle(label_0_indices)
    random.shuffle(label_1_indices)

    # Divide the indices into the final splits based on the sizes calculated above
    train_indices = label_0_indices[:num_train_0] + label_1_indices[:num_train_1]
    dev_indices = label_0_indices[num_train_0:num_train_0 + num_dev_0] + label_1_indices[
                                                                         num_train_1:num_train_1 + num_dev_1]
    test_indices = label_0_indices[num_train_0 + num_dev_0:] + label_1_indices[num_train_1 + num_dev_1:]
    print(train_indices)
    # Create the final datasets and dataloaders using the indices
    train_dataset = Subset(dataset_, train_indices)
    dev_dataset = Subset(dataset_, dev_indices)
    test_dataset = Subset(dataset_, test_indices)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    return train_dataloader, dev_dataloader, test_dataloader


# folder_path = "E:\\谣言检测深度学习工程\\微博数据集（9527条）所有特征\\ALL_textimage"
# train_dataloader, dev_dataloader, test_dataloader = set_up_data_loader(train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2)


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
        # outputs = model(text_features)
        print("train")
        logits = outputs[0]

        # 保存测试集结果

        print(logits)
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
            #outputs = model(text_features)
            print("eval")
            logits = outputs[0]
            print(logits)
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

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data_loader, desc="Iteration")):

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
            #outputs = model(text_features)
            print("test")
            logits = outputs[0]
            print(data_label)
            print(logits)

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

        eval_loss = eval_loss / nb_eval_steps
        preds = np.squeeze(preds)
        all_labels = np.squeeze(all_labels)

    return preds, all_labels, eval_loss


# 测试集分数
def test_score_model(model, test_data_loader, loss_fct, exclude_zero=False):
    predictions, y_test, test_loss = test_epoch(model, test_data_loader, loss_fct)

    predictions = predictions.round()

    f_score = f1_score(y_test, predictions, average="weighted")
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy, "F score:", f_score)
    return accuracy, f_score, test_loss


# 训练
def train(
        model,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        loss_fct,
):
    best_valid_loss = 9e+9
    run_name = str(wandb.run.id)
    valid_losses = []

    n_epochs = args.epochs
    for epoch_i in range(n_epochs):

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
        test_accuracy, test_f_score, test_loss = test_score_model(
            model, test_dataloader, loss_fct
        )

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_test_accuracy = test_accuracy
            best_valid_test_fscore = test_f_score

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
                "best_valid_test_fscore": best_valid_test_fscore

            }
        )


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
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def prep_for_training(num_training_steps, get_feature_data_item):
    dataset = get_appropriate_dataset(get_feature_data_item)
    text_model = dataset.tensors[1]
    GCN_model = dataset.tensors[2]
    image_model = dataset.tensors[3]
    ELA_model = dataset.tensors[4]

    model = DECM(text_model, GCN_model, image_model, ELA_model, args)
    #model = DECM(text_model, image_model, args)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    loss_fct = BCEWithLogitsLoss()

    # Prepare optimizer
    # used different learning rates for different componenets.
    '''
    if args.model == "DECM":

        GCN_params, image_params, ELA_params, other_params = model.get_params()
        optimizer_o, scheduler_o = get_optimizer_scheduler(other_params, num_training_steps,
                                                           learning_rate=args.learning_rate)
        optimizer_h, scheduler_h = get_optimizer_scheduler(ELA_params, num_training_steps,
                                                           learning_rate=args.learning_rate_h)
        optimizer_v, scheduler_v = get_optimizer_scheduler(image_params, num_training_steps,
                                                           learning_rate=args.learning_rate_v)
        optimizer_a, scheduler_a = get_optimizer_scheduler(GCN_params, num_training_steps,
                                                           learning_rate=args.learning_rate_a)

        optimizers = [optimizer_o, optimizer_h, optimizer_v, optimizer_a]
        schedulers = [scheduler_o, scheduler_h, scheduler_v, scheduler_a]

    else:
    '''
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


def main():
    wandb.init(project="DECM")
    wandb.config.update(args)

    if args.seed == -1:
        seed = random.randint(0, 9999)
        print("seed", seed)
    else:
        seed = args.seed

    wandb.config.update({"seed": seed}, allow_val_change=True)

    set_random_seed(seed)



    folder_path = "E:\\feature\\twitter\\new_twitter_list"
    text_features_wa = "E:\\feature\\twitter\\twitter_bert_sentence_embedding"
    GCN_features_wa = "E:\\feature\\twitter\\GCN_twittertext_feature"
    image_features_wa = "E:\\feature\\twitter\\twitter_pic_Resnet50"
    ELA_features_wa = "E:\\feature\\twitter\\twitter_pic_ELA_Resnet50"

    data_ite = data_dict(folder_path)

    get_feature_data_item = get_feature(data_ite,
                                        text_features_wa,
                                        GCN_features_wa,
                                        image_features_wa,
                                        ELA_features_wa)

    dataset_ = get_appropriate_dataset(get_feature_data_item)
    print(dataset_)
    # train_dataloader, dev_dataloader, test_dataloader = set_up_data_loader()

    train_dataloader, dev_dataloader, test_dataloader = set_up_data_loader(dataset_, train_ratio=0.7,
                                                                           dev_ratio=0.1, test_ratio=0.2)
    # print_dataloader(train_dataloader)
    # print_dataloader(dev_dataloader)
    # print_dataloader(test_dataloader)
    print(len(train_dataloader), len(dev_dataloader), len(test_dataloader))
    print("Dataset Loaded")

    num_training_steps = len(train_dataloader) * args.epochs

    model, optimizers, schedulers, loss_fct = prep_for_training(
        num_training_steps, get_feature_data_item
    )
    print("Model Loaded: ", args.model)
    train(
        model,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        optimizers,
        schedulers,
        loss_fct,
    )


if __name__ == "__main__":
    main()

'''
# 已知DECM模型输入为（Text_embadding, Image_embadding, Text_embadding_len, Image_embadding_len）

'''
