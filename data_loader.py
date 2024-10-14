import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

def load_and_prepare_data(file_path='ml-100k/u.data'):
    ratings = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    print(f"Dataset shape: {ratings.shape}")
    print(f"\nRatings distribution:\n{ratings['rating'].value_counts().sort_index()}")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    return ratings, trainset, testset