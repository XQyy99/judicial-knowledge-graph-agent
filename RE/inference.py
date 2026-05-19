# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification
import itertools

###########################################
#             配置与加载模型              #
###########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NER模型与标签（示例）
NER_MODEL_DIR = r"D:\dl\毕业设计\xxcq\o1\NER\ner_output\原标签\lr_3e-05_epo_10_bs_16_XL_512\checkpoint-710"
entity_types = ["Nh", "Ns", "NT", "NDR", "NW"]
ner_labels = ["O"] + ["B-"+t for t in entity_types] + ["I-"+t for t in entity_types]
ner_label2id = {label: i for i, label in enumerate(ner_labels)}
ner_id2label = {i: label for label, i in ner_label2id.items()}
TOKEN_DIR=r"D:\dl\bert-crf-token_classification_ner-master\pretrain\bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(TOKEN_DIR)
ner_model = BertForTokenClassification.from_pretrained(NER_MODEL_DIR).to(device)
ner_model.eval()

# RE模型与标签（示例）
RE_MODEL_DIR = r"D:\dl\毕业设计\xxcq\o1\RE\relation_extraction_output\bert\lr_3e-05_epo_10_bs_16_ML_512\checkpoint-3020"
RELATION_LABELS = ["sell_drugs_to", "traffic_in", "possess", "provide_shelter_for", "NA"]
re_label2id = {label: i for i, label in enumerate(RELATION_LABELS)}
re_id2label = {i: label for label, i in re_label2id.items()}
re_model = BertForSequenceClassification.from_pretrained(RE_MODEL_DIR).to(device)
re_model.eval()

###########################################
#              函数定义                   #
###########################################

def infer_entities(sentence: str, max_length=512):
    # 对输入句子进行编码
    encoding = tokenizer(sentence, return_offsets_mapping=True, truncation=True, 
                             padding="max_length", max_length=max_length, return_tensors='pt')
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()  # batch=1，所以取[0]

    with torch.no_grad():
        outputs = ner_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=2).squeeze(0).tolist()  # [seq_len]

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    sep_index = input_ids[0].tolist().index(tokenizer.sep_token_id)
    valid_start, valid_end = 1, sep_index - 1

    pred_labels = [ner_id2label[p] for p in preds]

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
                    "end_char": offset_mapping[i-1][1]
                })

            # 开始记录新实体
            current_entity_type = label[2:]  # 去掉B-
            current_entity = label
            current_start_char = start_char
            current_text_pieces = [sentence[start_char:end_char]]

        elif label.startswith("I-") and current_entity is not None and label[2:] == current_entity_type:
            current_text_pieces.append(sentence[start_char:end_char])
        else:
            # O或者类型不匹配，表示实体结束
            if current_entity is not None:
                # 保存上一个实体
                entities.append({
                    "text": "".join(current_text_pieces),
                    "type": current_entity_type,
                    "start_char": current_start_char,
                    "end_char": offset_mapping[i-1][1]
                })
                current_entity = None
                current_entity_type = None
                current_text_pieces = []

    # 如果最后一个token也在实体中，结束时保存
    if current_entity is not None:
        entities.append({
            "text": "".join(current_text_pieces),
            "type": current_entity_type,
            "start_char": current_start_char,
            "end_char": offset_mapping[valid_end][1]
        })

    return entities

def infer_relation(sentence: str, entity1: str, entity2: str, model, tokenizer, label2id, id2label, max_length=512):
    input_text = sentence + " [SEP] " + entity1 + " [SEP] " + entity2
    encodings = tokenizer(
        input_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()

    relation_label = id2label[pred_id]
    if relation_label == "NA":
        return None
    else:
        return relation_label

###########################################
#               主程序示例                #
###########################################
if __name__ == "__main__":
    user_input = "公诉机关指控：2021年1月30日22时许，被告人徐某在徐某某位于武汉市新洲区徐古街徐古菜场62号房屋中，以人民币300元的价格向刘某某贩卖3颗疑似毒品甲基苯丙胺片剂（俗称“麻果”），以人民币200元的价格向徐某某贩卖2颗疑似毒品甲基苯丙胺片剂。交易完成后，徐某被当场查获，办案民警从刘某某身上搜出3颗疑似毒品甲基苯丙胺片剂并予以扣押。徐某某将其购买的2颗疑似毒品甲基苯丙胺片剂扔弃在菜场走道路边，后民警沿途查找仍未找到。民警将扣押的毒品予以称量、取样并送检。经称量，搜出的3颗疑似毒品甲基苯丙胺片剂净重0.28克。经武汉市公安毒品司法鉴定中心鉴定，检材中均检出毒品甲基苯丙胺成分。次日，徐某被刑事拘留。公诉机关认为"


    # 1. NER识别实体
    result_entities = infer_entities(user_input)
    print("识别出的实体：")
    for ent in result_entities:
        print(f"实体: {ent['text']}, 类型: {ent['type']}, 起始: {ent['start_char']}, 结束: {ent['end_char']}")

    # 2. 对所有实体对进行RE预测
    # 如果只有一种类型实体，需要自行判断需要的实体组合规则。
    # 这里简单演示将所有识别出的实体两两组合
    relations = []
    for (e1, e2) in itertools.permutations(result_entities, 2):
        # e1, e2都是dict: {"text": ..., "type": ...}
        rel = infer_relation(user_input, e1['text'], e2['text'], re_model, tokenizer, re_label2id, re_id2label)
        if rel is not None:
            relations.append((e1['text'], e2['text'], rel))

    # 输出结果
    if relations:
        print("\n识别出的关系：")
        for (ent1, ent2, rel) in relations:
            print(f"实体'{ent1}' 与 实体'{ent2}' 之间的关系: {rel}")
    else:
        print("\n未检测到明确的关系。")