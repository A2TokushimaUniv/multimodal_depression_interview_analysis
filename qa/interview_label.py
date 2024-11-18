import argparse
import pandas as pd
from logzero import logger
from sklearn.model_selection import train_test_split

TRAIN_DATA_RATIO = 0.7
VALID_DATA_RATIO = 0.2
TEST_DATA_RATIO = 0.1
RANDOM_STATE = 42  # 乱数シードを設定して分割データの再現性を担保する


def _add_fold(results_df):
    """
    データを訓練データ、検証データ、テストデータに分割する
    """
    logger.info("データを分割します")
    # 層化抽出によりtrainとvalidationデータに分割
    train_df, temp_df = train_test_split(
        results_df,
        test_size=1 - TRAIN_DATA_RATIO,
        stratify=results_df["label"],
        random_state=RANDOM_STATE,
    )
    # 残りのデータをさらにvalidationとtestに分割
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_DATA_RATIO / (TEST_DATA_RATIO + VALID_DATA_RATIO),
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )
    # fold列の追加
    train_df["fold"] = "train"
    valid_df["fold"] = "valid"
    test_df["fold"] = "test"

    # 各データセットの情報を表示
    logger.info("訓練データ:")
    logger.info(f"  全データ数: {len(train_df)}")
    logger.info(
        f"  ラベル1（鬱状態）のデータ数: {train_df[train_df['label'] == 1].shape[0]}"
    )
    logger.info(
        f"  ラベル0（非鬱状態）のデータ数: {train_df[train_df['label'] == 0].shape[0]}"
    )
    logger.info("検証データ:")
    logger.info(f"  全データ数: {len(valid_df)}")
    logger.info(
        f"  ラベル1（鬱状態）のデータ数: {valid_df[valid_df['label'] == 1].shape[0]}"
    )
    logger.info(
        f"  ラベル0（非鬱状態）のデータ数: {valid_df[valid_df['label'] == 0].shape[0]}"
    )
    logger.info("テストデータ:")
    logger.info(f"  全データ数: {len(test_df)}")
    logger.info(
        f"  ラベル1（鬱状態）のデータ数: {test_df[test_df['label'] == 1].shape[0]}"
    )
    logger.info(
        f"  ラベル0（非鬱状態）のデータ数: {test_df[test_df['label'] == 0].shape[0]}"
    )

    # データフレームを再結合して一つのデータフレームにする
    results_df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    return results_df


def main(adult_results_path, child_results_path):
    logger.info(
        f"{adult_results_path}と{child_results_path}からモデル学習のためのラベルを生成しています..."
    )
    adult_df = pd.read_csv(adult_results_path)
    child_df = pd.read_csv(child_results_path)
    # ID列とPHQ9_Binary列を抽出して列名を変更
    adult_df = adult_df[["ID", "PHQ9_Binary"]].rename(
        columns={"ID": "index", "PHQ9_Binary": "label"}
    )
    child_df = child_df[["ID", "PHQ9_Binary"]].rename(
        columns={"ID": "index", "PHQ9_Binary": "label"}
    )
    # 縦方向に連結
    results_df = pd.concat([adult_df, child_df], axis=0).reset_index(drop=True)

    # 合計データ数、labelが1のデータ数、labelが0のデータ数を表示
    total_data = len(results_df)
    label_1_count = results_df[results_df["label"] == 1].shape[0]
    label_0_count = results_df[results_df["label"] == 0].shape[0]
    logger.info("データセット情報:")
    logger.info(f"  全データ数: {total_data}")
    logger.info(f"  ラベル1（鬱状態）のデータ数: {label_1_count}")
    logger.info(f"  ラベル0（非鬱状態）のデータ数: {label_0_count}")

    results_df = _add_fold(results_df)
    results_df.to_csv("interview_label.csv", index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adult_results_path", type=str, default="../data/qa/adult_results.csv"
    )
    parser.add_argument(
        "--child_results_path", type=str, default="../data/qa/child_results.csv"
    )
    args = parser.parse_args()
    main(args.adult_results_path, args.child_results_path)
