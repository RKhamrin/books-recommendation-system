import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder


def prepare_bert4rec_data(
    users_items_ratings: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    min_seq_len: int = 3,
) -> tuple:
    """
    Преобразует user_id, item_id, rating → BERT4Rec формат (sequences).
    АВТОМАТИЧЕСКИ генерирует timestamp: 1,2,3... для каждого пользователя.

    Args:
        users_items_ratings: DataFrame [user_col, item_col, rating_col]
        min_seq_len: минимальная длина последовательности пользователя

    Returns:
        tuple: (item_encoder, user_sequences, train_df, valid_df, test_df)
    """

    df = users_items_ratings[[user_col, item_col, rating_col]].copy()

    user_counts = df[user_col].value_counts()
    active_users = user_counts[user_counts >= min_seq_len].index
    df = df[df[user_col].isin(active_users)].copy()

    df = df.sort_values(user_col)
    df["timestamp"] = df.groupby(user_col).cumcount() + 1

    item_encoder = LabelEncoder()
    df["item_idx"] = item_encoder.fit_transform(df[item_col])
    n_items = len(item_encoder.classes_)

    df = df.sort_values([user_col, "timestamp"])

    user_sequences = defaultdict(list)
    for _, row in df.iterrows():
        user_sequences[row[user_col]].append(row["item_idx"])

    sequences = [seq for seq in user_sequences.values() if len(seq) >= min_seq_len]

    np.random.seed(42)
    np.random.shuffle(sequences)
    n = len(sequences)

    train_seqs = sequences[: int(0.8 * n)]
    valid_seqs = sequences[int(0.8 * n) : int(0.9 * n)]
    test_seqs = sequences[int(0.9 * n) :]

    def seqs_to_inter(seq_list, prefix):
        inter_data = []
        for i, seq in enumerate(seq_list):
            for j, item_id in enumerate(seq):
                inter_data.append([i, item_id, 1])
        return pd.DataFrame(inter_data, columns=["user_id", "item_id", "rating"])

    train_df = seqs_to_inter(train_seqs, "train")
    valid_df = seqs_to_inter(valid_seqs, "valid")
    test_df = seqs_to_inter(test_seqs, "test")

    return item_encoder, sequences, train_df, valid_df, test_df
