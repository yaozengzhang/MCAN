The data can be found in the related paper. To view the changes in various metrics during model training, you need to log in to WANDB in advance to access them.

The Ablation folders contain files for the variant networks of MCAN, used in ablation experiments.
The get_text_writing_style_feature folders include files for extracting text writing style features, achieved by pre-training a text-GCN. BERT_ResNet50_feature folder include files for extracting text semantic features and image semantic-physical features
Before training, four types of features should be stored in four separate folders.

MCAN_weibo_zt and MCAN_twitter_zt are the training files, while MCAN represents the network.

The text-GCN get_text_writing_style_feature project is based on the work of another author. Unfortunately, as it was from a long time ago, I am unable to identify the author. If the author sees this, please contact me so I can acknowledge your contribution in this project.

The current files are unorganized and contain messy content with Chinese comments. Further organization and refinement will be carried out later.
