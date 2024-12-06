import os
import torch
from transformers import BertTokenizer, BertModel

# from pytorch_pretrained_bert import BertTokenizer, BertModel
# 加载中文bert
# tokenizer = BertTokenizer.from_pretrained('D://bert-base-chinese')  # 加载base模型的对应的切词器
# model = BertModel.from_pretrained('D://bert-base-chinese')
# uncased_L-12_H-768_A-12
# 加载英文bert
tokenizer = BertTokenizer.from_pretrained('D://bert-base-cased-english')  # 加载base模型的对应的切词器
model = BertModel.from_pretrained('D://bert-base-cased-english', ignore_mismatched_sizes=True)


# print(tokenizer) # 打印出对应的信息，如base模型的字典大小，截断长度等等
# np.read()

def BERTMORE(textcontent=None, SEQ_LEN=None):
    if textcontent is None:
        textcontent = ""
    if SEQ_LEN is None:
        SEQ_LEN = len(textcontent)

    #    SEQ_LEN=length of textcontent
    token = tokenizer.tokenize(textcontent)  # 切词
    # print(token) # 切词结果
    indexes = tokenizer.convert_tokens_to_ids(token)  # 将词转换为对应字典的id
    # print(indexes) # 输出id
    tokens = tokenizer.convert_ids_to_tokens(indexes)  # 将id转换为对应字典的词
    # print(tokens) # 输出词

    # word_id_list = []
    split_tokens = token  # .tokenize(textcontent)
    if len(split_tokens) > SEQ_LEN:
        split_tokens = split_tokens[:SEQ_LEN]
    else:
        while len(split_tokens) < SEQ_LEN:
            split_tokens.append('[PAD]')

    if not split_tokens:  # if split_tokens is empty, add a [PAD] token
        split_tokens.append('[PAD]')

    # 使用这种方法对句子编码会自动添加[CLS] 和[SEP]
    input_ids = torch.tensor(tokenizer.encode(split_tokens)).unsqueeze(0)
    # print(input_ids)
    outputs = model(input_ids)
    cls_id = tokenizer._convert_token_to_id('[CLS]')
    # sep_id = tokenizer._convert_token_to_id('[SEP]')
    # print(cls_id, sep_id)
    #   print(tokens)  # 输出词
    sequence_output = outputs[0]
    #   print(sequence_output)
    #   print(sequence_output.shape)  ## 字向量
    #    torch.save(sequence_output, 'sequence_output.pkl')

    # aa=torch.load('sequence_output.pkl')
    # print(aa)

    return sequence_output


txt_folder = "C:\\Users\\Zhang15197663066\\Desktop\\谣言信息检测-博一工程\\twitter所有特征\\new_twitter_list"
output_folder = "C:\\Users\\Zhang15197663066\\Desktop\\谣言信息检测-博一工程\\twitter所有特征\\twitter_bert_sentence_embedding"

for filename in os.listdir(txt_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(txt_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            data_name = lines[0].strip()
            text_content = lines[2].strip()
            sentence_embedding = BERTMORE(text_content)
            sentence_vector = torch.sum(sentence_embedding, dim=1) / sentence_embedding.shape[1]
            output_filepath = os.path.join(output_folder, f"{data_name}.pt")
            torch.save(sentence_vector, output_filepath)
            print(f"{data_name}: {sentence_vector.shape}")
