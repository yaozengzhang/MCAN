# MCAN

## Paper Citation

Zhang, Y., Ma, J., & Jia, Y. (2025). MCAN: multimodal cross-aware network for fake news detection by extracting semantic-physical feature consistency. The Journal of Supercomputing, 81, Article 299. Springer. DOI: https://doi.org/10.1007/s11227-024-06815-1

MCAN is a multimodal rumor detection project. The model uses four pre-extracted features for each text-image sample:

- BERT text semantic feature: 768 dimensions
- TextGCN writing-style feature: 200 dimensions
- Original image ResNet50 feature: 1024 dimensions
- ELA image ResNet50 feature: 1024 dimensions

The training scripts are written for Twitter and Weibo experiments. They expect the four feature types to be prepared before training and stored in separate folders as `.pt` tensors.

## Project Structure

```text
MCAN/
├── MCAN.py
├── MCAN_twitter_zt.py
├── MCAN_weibo_zt.py
├── data_id/
├── Abolution/
├── BERT_RESNET50_feature/
└── get_text_writing_style_feature/
```

### Main Files

- `MCAN.py`: defines the MCAN network. The main class is `DECM`, which fuses BERT text, TextGCN, original image, and ELA image features through cross-attention layers and outputs a binary rumor prediction.
- `MCAN_twitter_zt.py`: Twitter training script. It loads prepared Twitter sample files, four feature folders, train/test split labels, initializes wandb, trains MCAN, and reports metrics.
- `MCAN_weibo_zt.py`: Weibo training script. It follows the same training structure as the Twitter script and uses the Weibo split list.
- `data_id/`: dataset split lists for Twitter and Weibo. These files store sample IDs, split markers, and labels according to the original dataset-source split used in the experiments.
- `Abolution/`: ablation-model files for removing or combining different modalities, such as BERT-only, image-only, without GCN, without ELA, and pairwise feature combinations.
- `BERT_RESNET50_feature/`: feature extraction scripts for BERT text features, original/ELA image ResNet50 features, and ELA image generation.
- `get_text_writing_style_feature/`: TextGCN-related code for text preprocessing, graph construction, GCN training, and writing-style feature generation.

## Model Structure

The main MCAN model in `MCAN.py` uses this feature flow:

```text
BERT text feature (768) + TextGCN feature (200)
        -> text-style cross attention

Original image ResNet50 feature (1024) + ELA image ResNet50 feature (1024)
        -> image-ELA cross attention

Text-style representation + image-ELA representation
        -> cross-modal attention
        -> fully connected classifier
        -> binary output
```

The output is a single logit for binary rumor detection.

## Data Sources

### Twitter

- Dataset: MediaEval 2016 Verifying Multimedia Use
- GitHub: https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2016
- Source repository: MKLab-ITI/image-verification-corpus
- Citation: Boididou, C., Papadopoulos, S., Zampoglou, M., Apostolidis, L., Papadopoulou, O., & Kompatsiaris, Y. (2018). Detection and visualization of misleading content on Twitter. International Journal of Multimedia Information Retrieval, 7(1), 71-86.
- Split source: Twitter train IDs are derived from `mediaeval2016/devset/posts.txt`, and test IDs are derived from `mediaeval2016/testset/posts_groundtruth.txt`. Labels are mapped as `real -> 0` and `fake -> 1`.

### Weibo

- Dataset: Weibo multimodal rumor detection dataset
- GitHub: https://github.com/wangzhuang1911/Weibo-dataset
- Source repository: wangzhuang1911/Weibo-dataset
- Citation: Jin, Z., Cao, J., Guo, H., Zhang, Y., & Luo, J. (2017). Multimodal Fusion with Recurrent Neural Networks for Rumor Detection on Microblogs. ACM Multimedia 2017, 795-816.

Empty texts, meaningless texts, and samples without image references were removed during data cleaning. The final sample counts may therefore differ from the original dataset statistics.

## Data Format

The training scripts read sample `.txt` files from a text-image sample folder. Each sample file is expected to contain four lines:

```text
data_id
text_content
image_id
label
```

For each `data_id`, the scripts load four feature tensors:

```text
bert_text_feature_folder/{data_id}.pt
gcn_text_feature_folder/{data_id}.pt
image_resnet50_feature_folder/{data_id}.pt
ela_resnet50_feature_folder/{data_id}.pt
```

The split files used by the training scripts contain sample IDs, train/valid/test markers, and labels. The Weibo script uses train/valid/test markers. The Twitter script uses the dataset split labels available in the prepared split file.

## Split Lists

The `data_id/` folder contains the split-list files used to connect sample IDs, feature tensors, labels, and train/validation/test partitions:

```text
data_id/
├── twitter_tvt_list.txt
└── weibo_tvt_list.txt
```

Each row is tab-separated:

```text
data_id    split    label
```

- `twitter_tvt_list.txt`: Twitter/MediaEval 2016 sample split list.
- `weibo_tvt_list.txt`: Weibo sample split list.

These files follow the dataset split definitions from the corresponding datasets listed in Data Sources. They are split and label lists derived from the dataset sources, not raw text, raw images, or generated feature tensors. Keep these lists aligned with the prepared sample files and the four feature folders during reproduction.

## Feature Preparation

Prepare features before running MCAN training.

1. Prepare the cleaned text-image sample files from the Twitter or Weibo dataset.
2. Generate BERT text features with `BERT_RESNET50_feature/Bert_sentence_embedding_tpl.py`.
3. Generate ELA images with `BERT_RESNET50_feature/imagetoELA.py`.
4. Generate ResNet50 features for original images and ELA images with `BERT_RESNET50_feature/Resnet50_extract_tpl.py`.
5. Generate TextGCN writing-style features with the scripts in `get_text_writing_style_feature/`.
6. Store the four feature types in separate folders and make sure each sample can be matched by `data_id`.

The TextGCN workflow follows the original TextGCN-style pipeline:

```text
raw text corpus
-> cleaned corpus
-> text graph
-> GCN training
-> 200-dimensional text graph feature
```

The TextGCN writing-style feature extraction code in this repository uses the Twitter dataset as the included example.

## Training

Install the required Python packages according to the scripts being used. The project depends on PyTorch, transformers, torchvision, scikit-learn, pandas, numpy, tqdm, wandb, networkx, imbalanced-learn, Pillow, and scikit-image.

Before training, update the local data and feature folder variables inside the selected training script so they point to the prepared dataset and feature folders.

Run Twitter training:

```bash
python MCAN_twitter_zt.py
```

Run Weibo training:

```bash
python MCAN_weibo_zt.py
```

The scripts initialize wandb runs:

```text
Twitter project: MTAP_twitter_zt_08
Weibo project: MTAP_weibo_zt
```

Log in to wandb before running if online experiment tracking is needed. The scripts can save model weights under `best_weights/` when weight saving is enabled.

## Test Results Figures

The following figures show the test-set results obtained by selecting the best checkpoint according to validation performance during training.

Twitter dataset result:

![Twitter dataset test results](assets/task1B_twitter.svg)

Weibo dataset result:

![Weibo dataset test results](assets/task1B_weibo.svg)

## Ablation Experiments

The `Abolution/` folder contains MCAN variants for ablation studies:

- `MCANwoBERT.py`
- `MCANwoGCN.py`
- `MCANwoimage.py`
- `MCANwoELA.py`
- `MCANonlyBERT.py`
- `MCANonlyimage.py`
- `MCAN_BERTandELA.py`
- `MCAN_BERTandGCN.py`
- `MCAN_BERTandimage.py`
- `MCAN_GCNandELA.py`
- `MCAN_GCNandimage.py`
- `MCAN_imageandELA.py`

These files keep the same general attention-based modeling style while changing the available input modalities.

## Notes

- Large raw datasets, images, and generated feature tensors are not stored in this repository.
- The training scripts preserve the original experiment flow and use precomputed `.pt` features.
- The effective loss function in the current Twitter and Weibo training scripts is `FocalLoss`.
- The TextGCN code in `get_text_writing_style_feature/` is based on another author's earlier implementation. The original author is not currently identified. Please contact the repository owner if attribution should be added.
