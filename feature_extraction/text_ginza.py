import os
from logzero import logger
import pandas as pd
import spacy
from collections import Counter
from utils import get_text_files

nlp = spacy.load("ja_ginza_electra")

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
    "CharPerMinutes": "CharPerMinutes",
    "WordPerMinutes": "WordPerMinutes",
}


def _get_top_frequent_words(counter, top_num=5):
    """
    頻出単語を取得する
    """
    return dict(counter.most_common(top_num))


def _calculate_speech_rate(texts, start_seconds, end_seconds):
    """
    1分間の文字数と単語数を計算する
    """
    assert len(texts) == len(start_seconds) == len(end_seconds)
    char_counts = [len(text) for text in texts]
    word_counts = []
    for text in texts:
        doc = nlp(text)
        words = [token.text for token in doc if token.pos_ != "SPACE"]
        word_counts.append(len(words))
    durations = [end - start for start, end in zip(start_seconds, end_seconds)]
    char_per_second = [
        char_count / duration for char_count, duration in zip(char_counts, durations)
    ]
    average_char_per_minutes = (sum(char_per_second) / len(char_per_second)) * 60
    word_per_second = [
        word_count / duration for word_count, duration in zip(word_counts, durations)
    ]
    average_word_per_minutes = (sum(word_per_second) / len(word_per_second)) * 60
    return average_char_per_minutes, average_word_per_minutes


def _count_negative_words(texts, negative_nouns, negative_verb_adj):
    """
    ネガティブ単語をカウントする
    """
    negative_noun_count = 0
    total_noun_count = 0
    negative_verb_count = 0
    total_verb_adj_count = 0
    negative_noun_counter = Counter()
    negative_verb_adj_counter = Counter()

    for text in texts:
        doc = nlp(text)
        for token in doc:
            # ネガティブな名詞をカウント
            if token.pos_ == "NOUN":
                total_noun_count += 1
                if token.lemma_ in negative_nouns:
                    negative_noun_count += 1
                    negative_noun_counter[token.lemma_] += 1
            # ネガティブな用言をカウント
            elif token.pos_ in ["VERB", "ADJ"]:
                total_verb_adj_count += 1
                if token.lemma_ in negative_verb_adj:
                    negative_verb_count += 1
                    negative_verb_adj_counter[token.lemma_] += 1

    top_negative_nouns = _get_top_frequent_words(negative_noun_counter)
    top_negative_verb_adj = _get_top_frequent_words(negative_verb_adj_counter)
    percentage_negative_nouns = (negative_noun_count / total_noun_count) * 100
    percentage_negative_verb_adj = (negative_verb_count / total_verb_adj_count) * 100
    return (
        negative_noun_count,
        negative_verb_count,
        percentage_negative_nouns,
        percentage_negative_verb_adj,
        top_negative_nouns,
        top_negative_verb_adj,
    )


def _count_positive_words(texts, positive_nouns, positive_verb_adj):
    """
    ポジティブ単語をカウントする
    """
    positive_noun_count = 0
    total_noun_count = 0
    positive_verb_count = 0
    total_verb_adj_count = 0
    positive_noun_counter = Counter()
    positive_verb_counter = Counter()

    for text in texts:
        doc = nlp(text)
        for token in doc:
            # ポジティブな名詞をカウント
            if token.pos_ == "NOUN":
                total_noun_count += 1
                if token.lemma_ in positive_nouns:
                    positive_noun_count += 1
                    positive_noun_counter[token.lemma_] += 1
            # ポジティブな用言をカウント
            elif token.pos_ in ["VERB", "ADJ"]:
                total_verb_adj_count += 1
                if token.lemma_ in positive_verb_adj:
                    positive_verb_count += 1
                    positive_verb_counter[token.lemma_] += 1
    top_positive_nouns = _get_top_frequent_words(positive_noun_counter)
    top_positive_verb_adj = _get_top_frequent_words(positive_verb_counter)

    percentage_positive_nouns = (positive_noun_count / total_noun_count) * 100
    percentage_positive_verb_adj = (positive_verb_count / total_verb_adj_count) * 100
    return (
        positive_noun_count,
        positive_verb_count,
        percentage_positive_nouns,
        percentage_positive_verb_adj,
        top_positive_nouns,
        top_positive_verb_adj,
    )


def _get_negative_nouns(nouns_file):
    """
    極性辞書からネガティブな名詞のみを読み取る
    """
    negative_nouns = []
    nouns_df = pd.read_csv(nouns_file, sep="\t", header=None)
    for _, row in nouns_df.iterrows():
        if row[1] == "n":
            negative_nouns.append(str(row[0]).strip())
    return negative_nouns


def _get_positive_nouns(nouns_file):
    """
    極性辞書からポジティブな名詞のみを読み取る
    """
    positive_nouns = []
    nouns_df = pd.read_csv(nouns_file, sep="\t", header=None)
    for _, row in nouns_df.iterrows():
        if row[1] == "p":
            positive_nouns.append(str(row[0]).strip())
    return positive_nouns


def _get_negative_verb_adj(verbs_file):
    """
    極性辞書からネガティブな用言のみを読み込む
    """
    negative_verbs = []
    verbs_df = pd.read_csv(verbs_file, sep="\t", header=None)
    for _, row in verbs_df.iterrows():
        if "ネガ" in row[0]:
            negative_verbs.append(str(row[1]).replace(" ", "").strip())
    return negative_verbs


def _get_positive_verb_adj(verbs_file):
    """
    極性辞書からポジティブな用言のみを読み込む
    """
    positive_verbs = []
    verbs_df = pd.read_csv(verbs_file, sep="\t", header=None)
    for _, row in verbs_df.iterrows():
        if "ポジ" in row[0]:
            positive_verbs.append(str(row[1]).replace(" ", "").strip())
    return positive_verbs


def _add_results(
    qa_result_df,
    data_id,
    negative_noun_count,
    negative_verb_adj_count,
    positive_nouns_count,
    positive_verb_adj_count,
    percentage_negative_nouns,
    percentage_negative_verb_adj,
    percentage_positive_nouns,
    percentage_positive_verb_adj,
    char_per_minutes,
    word_per_minutes,
):
    """
    結果をDataFrameに追加する
    """
    qa_result_df.loc[
        qa_result_df["ID"] == data_id, column_names["NegativeNounCount"]
    ] = negative_noun_count
    qa_result_df.loc[
        qa_result_df["ID"] == data_id,
        column_names["NegativeVerbAdjCount"],
    ] = negative_verb_adj_count
    qa_result_df.loc[
        qa_result_df["ID"] == data_id, column_names["NegativeWordCount"]
    ] = negative_noun_count + negative_verb_adj_count

    qa_result_df.loc[
        qa_result_df["ID"] == data_id, column_names["PositiveNounCount"]
    ] = positive_nouns_count
    qa_result_df.loc[
        qa_result_df["ID"] == data_id,
        column_names["PositiveVerbAdjCount"],
    ] = positive_verb_adj_count
    qa_result_df.loc[
        qa_result_df["ID"] == data_id, column_names["PositiveWordCount"]
    ] = positive_nouns_count + positive_verb_adj_count

    qa_result_df.loc[
        qa_result_df["ID"] == data_id,
        column_names["PercentagePositiveNouns"],
    ] = percentage_positive_nouns
    qa_result_df.loc[
        qa_result_df["ID"] == data_id,
        column_names["PercentagePositiveVerbAdj"],
    ] = percentage_positive_verb_adj
    qa_result_df.loc[
        qa_result_df["ID"] == data_id,
        column_names["PercentageNegativeNouns"],
    ] = percentage_negative_nouns
    qa_result_df.loc[
        qa_result_df["ID"] == data_id,
        column_names["PercentageNegativeVerbAdj"],
    ] = percentage_negative_verb_adj
    qa_result_df.loc[
        qa_result_df["ID"] == data_id,
        column_names["CharPerMinutes"],
    ] = char_per_minutes
    qa_result_df.loc[
        qa_result_df["ID"] == data_id,
        column_names["WordPerMinutes"],
    ] = word_per_minutes
    return qa_result_df


def _write_result(file_name, result_dict):
    """
    結果を書き込む
    """
    with open(file_name, "w") as f:
        f.writelines("\n".join(str(k) + "," + str(v) for k, v in result_dict.items()))


def analyze_text(adult_qa_df, child_qa_df, input_data_dir, output_data_dir):
    """
    GiNZAと極性辞書を使ってテキストを分析する
    """
    logger.info("テキストを分析しています....")
    text_files = get_text_files(os.path.join(input_data_dir, "subject_text"))
    if len(text_files) == 0:
        logger.error("テキストファイルが見つかりませんでした")
        raise ValueError("テキストファイルが見つかりませんでした")
    logger.info(f"{len(text_files)}個のテキストファイルを読み込みました")

    nouns_file = "./sentiment_polarity/名詞.tsv"
    verb_adj_file = "./sentiment_polarity/用言.tsv"
    negative_nouns = _get_negative_nouns(nouns_file)
    positive_nouns = _get_positive_nouns(nouns_file)
    negative_verb_adj = _get_negative_verb_adj(verb_adj_file)
    positive_verb_adj = _get_positive_verb_adj(verb_adj_file)

    all_negative_noun_counter = Counter()
    all_negative_verb_adj_counter = Counter()
    all_positive_noun_counter = Counter()
    all_positive_verb_adj_counter = Counter()

    for data_id, text_file in text_files:
        logger.info(f"{text_file}からテキストを分析しています....")
        text_data = pd.read_csv(text_file)
        texts = text_data["text"].tolist()
        start_seconds = text_data["start_seconds"].tolist()
        end_seconds = text_data["end_seconds"].tolist()
        char_per_minutes, word_per_minutes = _calculate_speech_rate(
            texts, start_seconds, end_seconds
        )
        (
            negative_noun_count,
            negative_verb_adj_count,
            percentage_negative_nouns,
            percentage_negative_verb_adj,
            top_negative_nouns,
            top_negative_verb_adj,
        ) = _count_negative_words(texts, negative_nouns, negative_verb_adj)
        all_negative_noun_counter.update(top_negative_nouns)
        all_negative_verb_adj_counter.update(top_negative_verb_adj)
        (
            positive_nouns_count,
            positive_verb_adj_count,
            percentage_positive_nouns,
            percentage_positive_verb_adj,
            top_positive_nouns,
            top_positive_verb_adj,
        ) = _count_positive_words(texts, positive_nouns, positive_verb_adj)
        all_positive_noun_counter.update(top_positive_nouns)
        all_positive_verb_adj_counter.update(top_positive_verb_adj)

        adult_qa_df = _add_results(
            adult_qa_df,
            data_id,
            negative_noun_count,
            negative_verb_adj_count,
            positive_nouns_count,
            positive_verb_adj_count,
            percentage_negative_nouns,
            percentage_negative_verb_adj,
            percentage_positive_nouns,
            percentage_positive_verb_adj,
            char_per_minutes,
            word_per_minutes,
        )

        child_qa_df = _add_results(
            child_qa_df,
            data_id,
            negative_noun_count,
            negative_verb_adj_count,
            positive_nouns_count,
            positive_verb_adj_count,
            percentage_negative_nouns,
            percentage_negative_verb_adj,
            percentage_positive_nouns,
            percentage_positive_verb_adj,
            char_per_minutes,
            word_per_minutes,
        )

    os.makedirs(os.path.join(output_data_dir, "text_ranking"), exist_ok=True)
    all_top_negative_nouns = _get_top_frequent_words(all_negative_noun_counter, 100)
    logger.info(
        f"全テキスト中で、ネガティブな名詞で最も出現頻度の高かった単語: {all_top_negative_nouns}"
    )
    _write_result(
        os.path.join(output_data_dir, "text_ranking", "all_top_negative_nouns.csv"),
        all_top_negative_nouns,
    )

    all_top_negative_verb_adj = _get_top_frequent_words(
        all_negative_verb_adj_counter, 100
    )
    logger.info(
        f"全テキスト中で、ネガティブな用言で最も出現頻度の高かった単語: {all_top_negative_verb_adj}"
    )
    _write_result(
        os.path.join(output_data_dir, "text_ranking", "all_top_negative_verb_adj.csv"),
        all_top_negative_verb_adj,
    )

    all_top_positive_nouns = _get_top_frequent_words(all_positive_noun_counter, 100)
    logger.info(
        f"全テキスト中で、ポジティブな名詞で最も出現頻度の高かった単語: {all_top_positive_nouns}"
    )
    _write_result(
        os.path.join(output_data_dir, "text_ranking", "all_top_positive_nouns.csv"),
        all_top_positive_nouns,
    )

    all_top_positive_verb_adj = _get_top_frequent_words(
        all_positive_verb_adj_counter, 100
    )
    logger.info(
        f"全テキスト中で、ポジティブな用言で最も出現頻度の高かった単語: {all_top_positive_verb_adj}"
    )
    _write_result(
        os.path.join(output_data_dir, "text_ranking", "all_top_positive_verb_adj.csv"),
        all_top_positive_verb_adj,
    )
    logger.info("テキスト分析を終了しました")
    return adult_qa_df, child_qa_df
