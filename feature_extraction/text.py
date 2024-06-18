import glob
import os
from logzero import logger
import pandas as pd
import spacy

nlp = spacy.load("ja_ginza")


def count_negative_words(texts, negative_nouns, negative_verbs):
    negative_noun_count = 0
    negative_verb_count = 0
    for text in texts:
        doc = nlp(text)
        for token in doc:
            if token.pos_ == "NOUN" and token.lemma_ in negative_nouns:
                logger.info(f"ネガティブ名詞：{token.lemma_}")
                negative_noun_count += 1
            elif token.pos_ in ["VERB", "ADJ"] and token.lemma_ in negative_verbs:
                logger.info(f"ネガティブ動詞：{token.lemma_}")
                negative_verb_count += 1
    logger.info(f"Negative noun count: {negative_noun_count}")
    logger.info(f"Negative verb count: {negative_verb_count}")
    return negative_noun_count, negative_verb_count


def get_negative_nouns(nouns_file):
    negative_nouns = []
    nouns_df = pd.read_csv(nouns_file, sep="\t", header=None)
    for _, row in nouns_df.iterrows():
        if row[1] == "n":
            negative_nouns.append(str(row[0]).strip())
    return negative_nouns


def get_negative_verbs(verbs_file):
    negative_verbs = []
    verbs_df = pd.read_csv(verbs_file, sep="\t", header=None)
    for _, row in verbs_df.iterrows():
        if "ネガ" in row[0]:
            negative_verbs.append(str(row[1]).replace(" ", "").strip())
    return negative_verbs


def analyze_ginza(before_sum_df):
    riko_text_files = glob.glob(
        os.path.join("../data/preprocessed_data/text/riko", "*", "*.csv"),
        recursive=True,
    )
    igaku_text_files = glob.glob(
        os.path.join("../data/preprocessed_data/text/igaku", "*", "*.csv"),
        recursive=True,
    )
    negative_nouns = get_negative_nouns("./sentiment_polarity/名詞.tsv")
    negative_verbs = get_negative_verbs("./sentiment_polarity/用言.tsv")

    for riko_text_file in riko_text_files:
        logger.info(f"Counting negative words from {riko_text_file}.")
        riko_texts = pd.read_csv(riko_text_file)
        texts = riko_texts.iloc[:, 2].tolist()
        id = riko_text_file.split("/")[-2]
        if int(id) < 10:
            target = f"riko0{id}"
        else:
            target = f"riko{id}"
        negative_noun_count, negative_verb_count = count_negative_words(
            texts, negative_nouns, negative_verbs
        )
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "N_Noun_count"] = (
            negative_noun_count
        )
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "N_verb_count"] = (
            negative_verb_count
        )
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "N_word_count"] = (
            negative_noun_count + negative_verb_count
        )

    for igaku_text_file in igaku_text_files:
        logger.info(f"Counting negative words from {igaku_text_file}.")
        igaku_texts = pd.read_csv(igaku_text_file)
        texts = igaku_texts.iloc[:, 2].tolist()
        id = igaku_text_file.split("/")[-2]
        target = f"psy_c_{id}"
        negative_noun_count, negative_verb_count = count_negative_words(
            texts, negative_nouns, negative_verbs
        )
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "N_noun_count"] = (
            negative_noun_count
        )
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "N_verb_count"] = (
            negative_verb_count
        )
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "N_word_count"] = (
            negative_noun_count + negative_verb_count
        )

    logger.info(before_sum_df)
    return before_sum_df
