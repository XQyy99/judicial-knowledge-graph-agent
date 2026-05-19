import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import numpy as np

# 已训练模型所在目录
MODEL_DIR = r"D:\dl\毕业设计\xxcq\o1\NER\ner_output\bert\lr_3e-05_epo_6_bs_16_XL_256\checkpoint-426"

# 定义标签集，与训练时一致
entity_types = ["Nh", "Ns", "NT", "NDR", "NW"]
labels = ["O"] + ["B-"+t for t in entity_types] + ["I-"+t for t in entity_types]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# 加载分词器和模型
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForTokenClassification.from_pretrained(MODEL_DIR)

def infer_entities(sentence: str, max_length=128):
    # 对输入句子进行编码
    encoding = tokenizer(sentence, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=max_length, return_tensors='pt')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"][0].tolist()  # batch=1，所以取[0]
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=2).squeeze(0).tolist()  # [seq_len]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    
    # 查找[SEP]的位置，CLS在0，SEP通常在句子后面
    sep_index = input_ids[0].tolist().index(tokenizer.sep_token_id)
    valid_start, valid_end = 1, sep_index - 1  # 有效文本区域

    # 将预测结果映射回实体
    # 排除CLS/SEP以及padding部分
    pred_labels = [id2label[p] for p in preds]
    
    # 重建实体：根据B-XXX和I-XXX序列还原完整实体
    entities = []
    current_entity = None
    current_entity_type = None
    current_start_char = None
    current_text_pieces = []

    for i in range(valid_start, valid_end+1):
        label = pred_labels[i]
        start_char, end_char = offset_mapping[i]

        if label.startswith("B-"):
            # 如果有正在记录的实体，先结束它
            if current_entity is not None:
                # 保存上一个实体
                entities.append({
                    "text": "".join(current_text_pieces),
                    "type": current_entity_type,
                    "start_char": current_start_char,
                    "end_char": offset_mapping[i-1][1]  # 上一个实体的结束位置
                })
            # 开始记录新实体
            current_entity_type = label[2:]  # 去掉B-
            current_entity = label
            current_start_char = start_char
            current_text_pieces = [sentence[start_char:end_char]]
        
        elif label.startswith("I-") and current_entity is not None and label[2:] == current_entity_type:
            # 同一实体的延续
            current_text_pieces.append(sentence[start_char:end_char])
        
        else:
            # O或者类型不匹配，表示实体结束
            if current_entity is not None:
                # 保存上一个实体
                entities.append({
                    "text": "".join(current_text_pieces),
                    "type": current_entity_type,
                    "start_char": current_start_char,
                    "end_char": offset_mapping[i-1][1]  # 实体结束位置
                })
                current_entity = None
                current_entity_type = None
                current_text_pieces = []
            # O标签不记录实体

    # 如果最后一个token也在实体中，结束时保存
    if current_entity is not None:
        entities.append({
            "text": "".join(current_text_pieces),
            "type": current_entity_type,
            "start_char": current_start_char,
            "end_char": offset_mapping[valid_end][1]
        })

    return entities

if __name__ == "__main__":
    # 用户输入
    user_input = "林某某于2014年8月7日早上7时在养猪场贩卖海洛因给徐某某。"
    result_entities = infer_entities(user_input)
    print("识别出的实体：")
    for ent in result_entities:
        print(f"实体: {ent['text']}, 类型: {ent['type']}, 起始: {ent['start_char']}, 结束: {ent['end_char']}")