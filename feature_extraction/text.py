import glob
import os
from logzero import logger
import pandas as pd
import spacy

nlp = spacy.load("ja_ginza")

column_names = {
    "NegativeNounCount": "Neg_Noun_Count",
    "PositiveNounCount": "Pos_Noun_Count",
    "NegativeVerbAdjCount": "Neg_VerbAdj_Count",
    "PositiveVerbAdjCount": "Pos_VerbAdj_Count",
    "PositiveWordCount": "Pos_Word_Count",
    "NegativeWordCount": "Neg_Word_Count",
    "PercentageNegativeNouns": "Per_Neg_Noun",
    "PercentagePositiveNouns": "Per_Pos_Noun",
    "PercentageNegativeVerbAdj": "Per_Neg_VerbAdj",
    "PercentagePositiveVerbAdj": "Per_Pos_VerbAdj",
}


def count_negative_words(texts, negative_nouns, negative_verb_adj):
    negative_noun_count = 0
    total_noun_count = 0
    negative_verb_count = 0
    total_verb_adj_count = 0
    for text in texts:
        doc = nlp(text)
        for token in doc:
            # ネガティブな名詞をカウント
            if token.pos_ == "NOUN":
                total_noun_count += 1
                if token.lemma_ in negative_nouns:
                    logger.info(f"Negative noun：{token.lemma_}")
                    negative_noun_count += 1
            # ネガティブな用言をカウント
            elif token.pos_ in ["VERB", "ADJ"]:
                total_verb_adj_count += 1
                if token.lemma_ in negative_verb_adj:
                    logger.info(f"Negative verb or adj：{token.lemma_}")
                    negative_verb_count += 1

    percentage_negative_nouns = (negative_noun_count / total_noun_count) * 100
    percentage_negative_verb_adj = (negative_verb_count / total_verb_adj_count) * 100
    logger.info(f"Negative noun count: {negative_noun_count}")
    logger.info(
        f"Percentage of negative nouns among all nouns: {percentage_negative_nouns}"
    )
    logger.info(f"Negative verb and adj count: {negative_verb_count}")
    logger.info(
        f"Percentage of negative verb and adj among all verb and adj: {percentage_negative_verb_adj}"
    )
    return (
        negative_noun_count,
        negative_verb_count,
        percentage_negative_nouns,
        percentage_negative_verb_adj,
    )


def count_positive_words(texts, positive_nouns, positive_verb_adj):
    positive_noun_count = 0
    total_noun_count = 0
    positive_verb_count = 0
    total_verb_adj_count = 0
    for text in texts:
        doc = nlp(text)
        for token in doc:
            # ポジティブな名詞をカウント
            if token.pos_ == "NOUN":
                total_noun_count += 1
                if token.lemma_ in positive_nouns:
                    logger.info(f"Positive noun：{token.lemma_}")
                    positive_noun_count += 1
            # ポジティブな用言をカウント
            elif token.pos_ in ["VERB", "ADJ"]:
                total_verb_adj_count += 1
                if token.lemma_ in positive_verb_adj:
                    logger.info(f"Positive verb or adj：{token.lemma_}")
                    positive_verb_count += 1

    percentage_positive_nouns = (positive_noun_count / total_noun_count) * 100
    percentage_positive_verb_adj = (positive_verb_count / total_verb_adj_count) * 100
    logger.info(f"Positive noun count: {positive_noun_count}")
    logger.info(
        f"Percentage of positive nouns among all nouns: {percentage_positive_nouns}"
    )
    logger.info(f"Positive verb and adj count: {positive_verb_count}")
    logger.info(
        f"Percentage of positive verb and adj among all verb and adj: {percentage_positive_verb_adj}"
    )
    return (
        positive_noun_count,
        positive_verb_count,
        percentage_positive_nouns,
        percentage_positive_verb_adj,
    )


def get_negative_nouns(nouns_file):
    negative_nouns = []
    nouns_df = pd.read_csv(nouns_file, sep="\t", header=None)
    # 極性辞書からネガティブな名詞のみを読み取る
    for _, row in nouns_df.iterrows():
        if row[1] == "n":
            negative_nouns.append(str(row[0]).strip())
    return negative_nouns


def get_positive_nouns(nouns_file):
    positive_nouns = []
    nouns_df = pd.read_csv(nouns_file, sep="\t", header=None)
    # 極性辞書からポジティブな名詞のみを読み取る
    for _, row in nouns_df.iterrows():
        if row[1] == "p":
            positive_nouns.append(str(row[0]).strip())
    return positive_nouns


def get_negative_verb_adj(verbs_file):
    negative_verbs = []
    verbs_df = pd.read_csv(verbs_file, sep="\t", header=None)
    # 極性辞書からネガティブな用言のみを読み込む
    for _, row in verbs_df.iterrows():
        if "ネガ" in row[0]:
            negative_verbs.append(str(row[1]).replace(" ", "").strip())
    return negative_verbs


def get_positive_verb_adj(verbs_file):
    positive_verbs = []
    verbs_df = pd.read_csv(verbs_file, sep="\t", header=None)
    # 極性辞書からポジティブな用言のみを読み込む
    for _, row in verbs_df.iterrows():
        if "ポジ" in row[0]:
            positive_verbs.append(str(row[1]).replace(" ", "").strip())
    return positive_verbs


def add_results(
    qa_result_df,
    target,
    negative_noun_count,
    negative_verb_adj_count,
    positive_nouns_count,
    positive_verb_adj_count,
    percentage_negative_nouns,
    percentage_negative_verb_adj,
    percentage_positive_nouns,
    percentage_positive_verb_adj,
):
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["NegativeNounCount"]
    ] = negative_noun_count
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target,
        column_names["NegativeVerbAdjCount"],
    ] = negative_verb_adj_count
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["NegativeWordCount"]
    ] = negative_noun_count + negative_verb_adj_count

    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["PositiveNounCount"]
    ] = positive_nouns_count
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target,
        column_names["PositiveVerbAdjCount"],
    ] = positive_verb_adj_count
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["PositiveWordCount"]
    ] = positive_nouns_count + positive_verb_adj_count

    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target,
        column_names["PercentagePositiveNouns"],
    ] = percentage_positive_nouns
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target,
        column_names["PercentagePositiveVerbAdj"],
    ] = percentage_positive_verb_adj
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target,
        column_names["PercentageNegativeNouns"],
    ] = percentage_negative_nouns
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target,
        column_names["PercentageNegativeVerbAdj"],
    ] = percentage_negative_verb_adj
    return qa_result_df


def analyze_text(qa_result_df):
    riko_text_files = glob.glob(
        os.path.join("../data/preprocessed_data/text/riko", "*", "*.csv"),
        recursive=True,
    )
    igaku_text_files = glob.glob(
        os.path.join("../data/preprocessed_data/text/igaku", "*", "*.csv"),
        recursive=True,
    )
    nouns_file = "./sentiment_polarity/名詞.tsv"
    verb_adj_file = "./sentiment_polarity/用言.tsv"
    negative_nouns = get_negative_nouns(nouns_file)
    positive_nouns = get_positive_nouns(nouns_file)
    negative_verb_adj = get_negative_verb_adj(verb_adj_file)
    positive_verb_adj = get_positive_verb_adj(verb_adj_file)

    for riko_text_file in riko_text_files:
        logger.info(f"Counting negative words from {riko_text_file}....")
        riko_texts = pd.read_csv(riko_text_file)
        texts = riko_texts.iloc[:, 2].tolist()
        id = riko_text_file.split("/")[-2]
        if int(id) < 10:
            target = f"riko0{id}"
        else:
            target = f"riko{id}"
        (
            negative_noun_count,
            negative_verb_adj_count,
            percentage_negative_nouns,
            percentage_negative_verb_adj,
        ) = count_negative_words(texts, negative_nouns, negative_verb_adj)
        (
            positive_nouns_count,
            positive_verb_adj_count,
            percentage_positive_nouns,
            percentage_positive_verb_adj,
        ) = count_positive_words(texts, positive_nouns, positive_verb_adj)

        qa_result_df = add_results(
            qa_result_df,
            target,
            negative_noun_count,
            negative_verb_adj_count,
            positive_nouns_count,
            positive_verb_adj_count,
            percentage_negative_nouns,
            percentage_negative_verb_adj,
            percentage_positive_nouns,
            percentage_positive_verb_adj,
        )

    for igaku_text_file in igaku_text_files:
        logger.info(f"Counting negative words from {igaku_text_file}....")
        igaku_texts = pd.read_csv(igaku_text_file)
        texts = igaku_texts.iloc[:, 2].tolist()
        id = igaku_text_file.split("/")[-2]
        target = f"psy_c_{id}"
        (
            negative_noun_count,
            negative_verb_adj_count,
            percentage_negative_nouns,
            percentage_negative_verb_adj,
        ) = count_negative_words(texts, negative_nouns, negative_verb_adj)
        (
            positive_nouns_count,
            positive_verb_adj_count,
            percentage_positive_nouns,
            percentage_positive_verb_adj,
        ) = count_positive_words(texts, positive_nouns, positive_verb_adj)

        qa_result_df = add_results(
            qa_result_df,
            target,
            negative_noun_count,
            negative_verb_adj_count,
            positive_nouns_count,
            positive_verb_adj_count,
            percentage_negative_nouns,
            percentage_negative_verb_adj,
            percentage_positive_nouns,
            percentage_positive_verb_adj,
        )
    return qa_result_df
