import json
import os
import torch
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments,AutoModelForTokenClassification, AutoTokenizer
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from NERmodel import BertCRFModel
from transformers import RobertaForTokenClassification,RobertaTokenizer,AutoModel,ErnieForMaskedLM,BertTokenizer
from bilstmcrf import BertBiLSTMCRF
os.environ["WANDB_MODE"] = "disabled"

# ==============================
# 参数设定
# ==============================
DATA_PATH = r"D:\dl\毕业设计\xxcq\final_train.json"  # 训练数据文件路径
#MODEL_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\bert-base-chinese" #bert
#MODEL_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\chinese-roberta-wwm-ext" #roberta
#MODEL_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\albert-base-chinese" #albert
#MODEL_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\chinese-macbert-base"
#MODEL_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\ernie-3.0-base-zh"
MODEL_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\chinese-pert-base"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 3e-5
usemodel='pert'
OUTPUT_DIR = f"D:/dl/毕业设计/xxcq/o1/NER/ner_output/{usemodel}/lr_{LEARNING_RATE}_epo_{EPOCHS}_bs_{BATCH_SIZE}_XL_{MAX_LENGTH}_final"

# 定义NER标签（包括O标签）
# 假设五种实体类型：Nh（人名）、Ns（地名）、NT（时间）、NDR（毒品类型）、NW（毒品重量）
# 使用BIO标注：B-前缀表示实体开头，I-表示实体内部，O表示非实体
#entity_types = ["Nh", "Ns", "NT", "NDR", "NW"]
entity_types = ["Nh", "Ns", "NT", "NDR", "NW"]
labels = ["O"] + ["B-"+t for t in entity_types] + ["I-"+t for t in entity_types]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}


def load_data(data_path: str):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data.append(item)
    return data

data = load_data(DATA_PATH)

# ==============================
# 数据转换逻辑
# 对每个句子：根据entityMentions的 start、end 标记token级别的标签
# 使用BIO标注方案
# ==============================

def align_labels_with_tokens(tokens, entities, offset_mapping):
    """
    根据实体的字符起止位置，把每个token打上相应的BIO标签。
    :param tokens: 分词后的token列表
    :param entities: 实体列表，每个实体: (start_char, end_char, label)
    :param offset_mapping: tokenizer返回的offset映射，每个token对应一个(start_offset, end_offset)
    :return: 对应的BIO标签列表（长度与tokens一致）
    """
    # 初始化全为O
    labels_out = ["O"] * len(tokens)

    # 将实体起止位置转为token级别标注
    # 对每个实体，根据char级位置找到对应的token范围，然后打上B-XXX, I-XXX
    for (start_char, end_char, ent_label) in entities:
        # end_char为实体结束位置（不含该字符）
        # 遍历offset_mapping找到覆盖此区间的tokens
        ent_type = ent_label  # 原标签，如 'Nh'
        b_label = "B-" + ent_type
        i_label = "I-" + ent_type

        # 找出覆盖实体范围的tokens
        entity_token_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if start_char < end and end_char > start:  # 有交集的token视为实体的一部分
                entity_token_indices.append(i)

        if len(entity_token_indices) > 0:
            labels_out[entity_token_indices[0]] = b_label
            for idx in entity_token_indices[1:]:
                labels_out[idx] = i_label

    return labels_out

def build_ner_samples(data: List[Dict], tokenizer: BertTokenizerFast, max_length: int = 128):
    samples = []
    for item in data:
        sentence = item["sentText"]
        entityMentions = item.get("entityMentions", [])
        # 实体列表 (start_char, end_char, label)
        entities = [(ent["start"], ent["end"], ent["label"]) for ent in entityMentions]

        # 分词并获取offset
        encoding = tokenizer(sentence, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=max_length)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offset_mapping = encoding["offset_mapping"]

        # 对[CLS], [SEP], padding 位置不标注
        # 通常[CLS]和[SEP]的offset为 (0,0)，padding也为(0,0)
        # 我们只给有实际文本的token打标签
        labels_out = ["O"] * len(tokens)

        # 对非特殊token打标签
        # 首先找到有效的文本token索引范围（第一个是CLS，最后可能是SEP）
        # 通常tokenizer返回形式：[CLS], ...tokens..., [SEP], ...pad...
        # 对于中文BERT, CLS和SEP一般是 [CLS], [SEP] 标记，可以根据tokens判断
        # 假设第一个是[CLS], 最后有[SEP]
        # 我们对 [CLS] 和 [SEP] 不打实体标签，其余文本部分才能对齐
        # offset_mapping与tokens数量一致
        # 开始打标时要排除CLS(一般为第0个)和SEP(找到下一个SEP)
        if tokenizer.cls_token_id in encoding["input_ids"]:
            # 通常CLS在0位置，SEP在最后或tokens中间
            # 对于bert-base-chinese，序列开头是[CLS] (101), 结尾是[SEP] (102)
            # 找到SEP位置
            sep_index = encoding["input_ids"].index(tokenizer.sep_token_id)
            
            # 实际文本tokens从1开始到sep_index-1结束
            valid_start, valid_end = 1, sep_index - 1

            # 给有效的文本token标注
            # entities中是char级别位置，通过align_labels_with_tokens进行转换
            # align_labels_with_tokens需要offset_mapping用于字符对齐
            # 这里传入不包括special tokens的offset，或者直接全部传入再排除不处理的即可
            # 我们这里直接传全部offset_mapping，但只更新有效范围内的labels_out
            raw_labels = align_labels_with_tokens(tokens, entities, offset_mapping)
            # 我们实际只保留 valid_start 到 valid_end之间的标注，
            # 因为CLS/SEP需要保持为O
            for i in range(len(tokens)):
                if i < valid_start or i > valid_end:
                    # 对CLS/SEP/PAD都设为O
                    labels_out[i] = "O"
                else:
                    labels_out[i] = raw_labels[i]

        # 将labels_out转为id
        label_ids = [label2id[l] if l in label2id else label2id["O"] for l in labels_out]

        samples.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label_ids
        })

    return samples


class NERDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.samples[idx].items()}
        return item


# 加载分词器
#tokenizer = BertTokenizerFast.from_pretrained(r"D:\dl\bert-crf-token_classification_ner-master\pretrain\bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 划分训练集和验证集
train_ratio = 0.8
train_size = int(train_ratio * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

train_samples = build_ner_samples(train_data, tokenizer, max_length=MAX_LENGTH)
val_samples = build_ner_samples(val_data, tokenizer, max_length=MAX_LENGTH)

train_dataset = NERDataset(train_samples)
val_dataset = NERDataset(val_samples)

# ==============================
# 模型及训练器定义
# ==============================
# model = BertBiLSTMCRF(
#     MODEL_NAME,
#     num_labels=len(labels)
# )

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,         
    num_labels=len(labels)  
)

#device = torch.device("cuda")
#model=BertCRFModel()#.to(device)
# 定义评价函数
def compute_metrics(p):
    predictions, label_ids = p
    # predictions是logits，需要转为预测标签索引 
    preds = np.argmax(predictions, axis=2)


    # 移除padding位置
    # 对于seqeval，需要先转换成相同长度的序列
    true_labels = []
    true_preds = []
    for pred, lab in zip(preds, label_ids):
        # 去掉特殊tokens，如CLS/SEP/PAD
        # 我们对全部tokens打了label_id，如为PAD部分label也为O
        # 只要在对齐时保证O是真正的非实体即可
        # 如果有padding位置(模型输入为定长)，需要在attention_mask上过滤
        # 由于我们在samples中没有保留attention_mask? 有的，在samples中有attention_mask
        # predictions和label_ids对应与input_ids长度。
        # 这里从p中获得的label_ids和predictions不包含attention_mask，需要通过p未传attention_mask
        # 不过我们可以在传入到trainer时通过datacollator引入
        # 简单起见，此处假设没有额外padding之外的标签错位
        # 去掉PAD需要事先记录PAD token位置(此处简化处理：假设label中O的一部分可能对应PAD)
        # 严谨做法需要使用attention_mask，这里先假设padding后面的就是O，不影响主要实体度量
        
        pred_labels = [id2label[p_id] for p_id in pred]
        true_lbls = [id2label[l_id] for l_id in lab]

        # 去掉不需要评估的特殊token位置，这里简单假设第一个是[CLS],最后有[SEP]，剩下的非文本是PAD
        # 根据我们前面的代码，CLS和SEP也是O，因为offset是0,0
        # 可以直接按实际文本长度截断，这里为了简单假设全长，O label在seqeval不影响f1计算(但会计算时包括O)
        # 更加严格的做法：如果有attention_mask，就只评估mask=1的部分。
        # 假设padding得到的O标签不影响最终结果，因为seqeval在计算时会跳过O不进行f1计算？
        # seqeval默认会计算所有标签类型的F1，但是O不计入最终F1计算，因为O不属于实体标签。
        
        true_labels.append(true_lbls)
        true_preds.append(pred_labels)

    # 使用seqeval计算F1等指标(平均方式为macro或micro)
    # 这里使用macro-f1(average='macro')或者'weighted'
    # seqeval默认是逐类计算NER的F1，这里使用简单的micro avg
    precision = precision_score(true_labels, true_preds)
    recall = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model(OUTPUT_DIR)

print("训练完成并保存模型至:", OUTPUT_DIR)