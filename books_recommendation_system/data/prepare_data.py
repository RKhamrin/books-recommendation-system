import pandas as pd 
import numpy as np

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

def prepare_bert4rec_data(users_items_ratings: pd.DataFrame, 
                         user_col: str = 'user_id', 
                         item_col: str = 'item_id', 
                         rating_col: str = 'rating',
                         min_seq_len: int = 3) -> tuple:
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
    
    # Фильтр по минимальной активности
    user_counts = df[user_col].value_counts()
    active_users = user_counts[user_counts >= min_seq_len].index
    df = df[df[user_col].isin(active_users)].copy()
    
    print(f"Active users: {len(active_users)} из {len(user_counts)}")
    
    # 🔥 АВТО-ГЕНЕРАЦИЯ TIMESTAMP по группам пользователей
    df = df.sort_values(user_col)  # сначала группируем по user
    df['timestamp'] = df.groupby(user_col).cumcount() + 1  # 1,2,3... для каждого user
    
    # Энкодинг items
    item_encoder = LabelEncoder()
    df['item_idx'] = item_encoder.fit_transform(df[item_col])
    n_items = len(item_encoder.classes_)
    
    print(f"Total items: {n_items}")
    
    # 🔥 СОРТИРОВКА ПО TIMESTAMP для хронологии
    df = df.sort_values([user_col, 'timestamp'])
    
    # Создание последовательностей
    user_sequences = defaultdict(list)
    for _, row in df.iterrows():
        user_sequences[row[user_col]].append(row['item_idx'])
    
    # Конвертация в список последовательностей
    sequences = [seq for seq in user_sequences.values() if len(seq) >= min_seq_len]
    
    print(f"Final sequences: {len(sequences)}, Avg len: {np.mean([len(s) for s in sequences]):.1f}")
    
    # Split: 8:1:1 (train/valid/test)
    np.random.seed(42)
    np.random.shuffle(sequences)
    n = len(sequences)
    
    train_seqs = sequences[:int(0.8*n)]
    valid_seqs = sequences[int(0.8*n):int(0.9*n)]
    test_seqs = sequences[int(0.9*n):]
    
    # RecBole формат (.inter файлы)
    def seqs_to_inter(seq_list, prefix):
        inter_data = []
        for i, seq in enumerate(seq_list):
            for j, item_id in enumerate(seq):
                inter_data.append([i, item_id, 1])  # user_id, item_id, rating=1 (implicit)
        return pd.DataFrame(inter_data, columns=['user_id', 'item_id', 'rating'])
    
    train_df = seqs_to_inter(train_seqs, 'train')
    valid_df = seqs_to_inter(valid_seqs, 'valid')
    test_df = seqs_to_inter(test_seqs, 'test')
    
    return item_encoder, sequences, train_df, valid_df, test_df