from transformers import BertTokenizer, BertModel
import torch


def get_bert_feature(text):
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
    # 文章をトークン化
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # BERTモデルに入力して特徴量を抽出
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state
    # 最後の隠れ層を取得し、平均を計算
    features = torch.mean(last_hidden_states, dim=1).squeeze().numpy()
    return features
