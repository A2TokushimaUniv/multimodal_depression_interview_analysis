# # TODO: GiNZAに変更する
# def get_bert_feature(text):
#     tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
#     model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
#     tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

#     # BERTの最大トークン数を取得
#     max_token_length = model.config.max_position_embeddings

#     # 文章を最大トークン数ごとにセグメントに分割
#     # TODO: もっといいやり方があるかも？
#     segments = []
#     for i in range(0, tokenized_text.input_ids.size(1), max_token_length):
#         segment = {
#             key: value[:, i : i + max_token_length]
#             for key, value in tokenized_text.items()
#         }
#         segments.append(segment)

#     # セグメントごとに特徴量を抽出し、結合する
#     features = []
#     for segment in segments:
#         with torch.no_grad():
#             outputs = model(**segment)
#         features.append(outputs.last_hidden_state)

#     # 特徴量を結合して最終的な特徴量を得る
#     final_features = torch.cat(features, dim=1)
#     return final_features
