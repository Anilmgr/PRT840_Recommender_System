"""
Simple Recommender System with Attack Analysis
============================================

A recommender system implementation that evaluates the performance of different
collaborative filtering approaches (User-based CF, Item-based CF, and SVD) under
various attack scenarios.

Dependencies:
    - pandas
    - numpy
    - surprise
    - tabulate

The system uses the MovieLens 100K dataset and evaluates models using metrics
like RMSE, MAE, Hit Rate, ARHR, MAP@K, and NDCG@K.
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
from tabulate import tabulate
from typing import Dict, List, Optional, Tuple, Any


def load_data(filepath: str) -> Tuple[Dataset, pd.DataFrame]:
    """
    Load and prepare the MovieLens dataset.
    
    Args:
        filepath: Path to the MovieLens data file
        
    Returns:
        Tuple of (Surprise Dataset, pandas DataFrame)
    """
    # Load the MovieLens dataset
    ratings = pd.read_csv(
        filepath,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    print(f"Dataset shape: {ratings.shape}")
    print(f"\nRatings distribution:\n{ratings['rating'].value_counts().sort_index()}")
    
    # Prepare data for Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings[['user_id', 'item_id', 'rating']],
        reader
    )
    
    return data, ratings


def generate_random_attack(
    num_fake_users: int,
    num_items: int,
    target_item: int,
    filler_size: float = 0.1
) -> np.ndarray:
    """
    Generate random attack profiles.
    
    Args:
        num_fake_users: Number of fake users to generate
        num_items: Total number of items in the dataset
        target_item: The item to be pushed
        filler_size: Proportion of items to be rated by each fake user
        
    Returns:
        Array of attack profiles
    """
    attack_profiles = []
    for _ in range(num_fake_users):
        profile = np.random.randint(1, 6, size=int(num_items * filler_size))
        profile = np.pad(profile, (0, num_items - len(profile)), 'constant')
        profile[target_item] = 5  # Set target item rating to maximum
        attack_profiles.append(profile)
    return np.array(attack_profiles)


def generate_average_attack(
    num_fake_users: int,
    num_items: int,
    target_item: int,
    ratings_data: pd.DataFrame,
    filler_size: float = 0.1
) -> np.ndarray:
    """
    Generate average attack profiles.
    
    Args:
        num_fake_users: Number of fake users to generate
        num_items: Total number of items in the dataset
        target_item: The item to be pushed
        ratings_data: The original ratings data
        filler_size: Proportion of items to be rated by each fake user
        
    Returns:
        Array of attack profiles
    """
    item_means = ratings_data.groupby('item_id')['rating'].mean()
    attack_profiles = []
    for _ in range(num_fake_users):
        profile = item_means.sample(n=int(num_items * filler_size)).round().values
        profile = np.pad(profile, (0, num_items - len(profile)), 'constant')
        profile[target_item] = 5  # Set target item rating to maximum
        attack_profiles.append(profile)
    return np.array(attack_profiles)


def dcg_score(relevance: List[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG) at rank k.
    
    Args:
        relevance: List of relevance scores
        k: Rank to calculate DCG at
        
    Returns:
        DCG score
    """
    relevance = np.asarray(relevance)[:k]
    if relevance.size:
        return np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))
    return 0.0


def ndcg_score(relevance: List[int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.
    
    Args:
        relevance: List of relevance scores
        k: Rank to calculate NDCG at
        
    Returns:
        NDCG score
    """
    best = sorted(relevance, reverse=True)
    dcg = dcg_score(relevance, k)
    idcg = dcg_score(best, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    model: Any,
    testset: List,
    attack_profiles: Optional[np.ndarray] = None,
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate a model's performance with various metrics.
    
    Args:
        model: The recommender model to evaluate
        testset: The test set to evaluate on
        attack_profiles: Attack profiles to inject (if any)
        k: The k value for top-k metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    if attack_profiles is not None:
        # Create a new dataset that includes both original ratings and attack profiles
        original_ratings = [
            (model.trainset.to_raw_uid(u), model.trainset.to_raw_iid(i), r)
            for (u, i, r) in model.trainset.all_ratings()
        ]
        
        attack_ratings = [
            (f'fake_user_{i}', j, rating)
            for i, profile in enumerate(attack_profiles)
            for j, rating in enumerate(profile)
            if rating != 0
        ]
        
        all_ratings = original_ratings + attack_ratings
        df = pd.DataFrame(all_ratings, columns=['user', 'item', 'rating'])
        
        reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
        data = Dataset.load_from_df(df, reader)
        
        # Build a new trainset and fit the model
        new_trainset = data.build_full_trainset()
        model.fit(new_trainset)
    
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Calculate ranking metrics
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r, iid))
    
    metrics = {
        'hit_count': 0,
        'arhr_sum': 0,
        'ap_sum': 0,
        'ndcg_sum': 0
    }
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        
        relevance = [1 if true_r >= 4 else 0 for _, true_r, _ in top_k]
        
        if 1 in relevance:
            metrics['hit_count'] += 1
            rank = relevance.index(1) + 1
            metrics['arhr_sum'] += 1 / rank
        
        ap = sum(
            sum(relevance[:i+1]) / (i+1) if rel else 0
            for i, rel in enumerate(relevance)
        ) / k
        metrics['ap_sum'] += ap
        
        ndcg = ndcg_score(relevance, k)
        metrics['ndcg_sum'] += ndcg
    
    num_users = len(user_est_true)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        f'Hit Rate@{k}': metrics['hit_count'] / num_users,
        f'ARHR@{k}': metrics['arhr_sum'] / num_users,
        f'MAP@{k}': metrics['ap_sum'] / num_users,
        f'NDCG@{k}': metrics['ndcg_sum'] / num_users
    }


def main():
    """Main execution function."""
    # Load and prepare data
    data, ratings = load_data('ml-100k/u.data')
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'UBCF': KNNBasic(sim_options={'user_based': True}),
        'IBCF': KNNWithMeans(sim_options={'user_based': False}),
        'SVD': SVD()
    }
    
    # Train models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(trainset)
    
    # Generate attack profiles
    num_items = ratings['item_id'].nunique()
    target_item = np.random.randint(0, num_items)
    num_fake_users = int(0.01 * ratings['user_id'].nunique())  # 1% of total users
    
    print("Generating attack profiles...")
    attack_scenarios = [
        ('No Attack', None),
        ('Random Attack', generate_random_attack(
            num_fake_users, num_items, target_item
        )),
        ('Average Attack', generate_average_attack(
            num_fake_users, num_items, target_item, ratings
        ))
    ]
    print("Attack profiles generated.")
    
    # Evaluate models
    results = {}
    for model_name, model in models.items():
        model_results = {}
        for scenario, attack_profiles in attack_scenarios:
            print(f"\nEvaluating {model_name} - {scenario}")
            model_results[scenario] = evaluate_model(model, testset, attack_profiles)
        results[model_name] = model_results
    
    # Prepare results table
    table_data = [
        [model_name, scenario] + list(metrics.values())
        for model_name, scenarios in results.items()
        for scenario, metrics in scenarios.items()
    ]
    
    headers = ["Model", "Scenario"] + list(
        next(iter(results.values()))["No Attack"].keys()
    )
    
    # Print and save results
    print("\nResults:")
    print(tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="grid"))
    
    results_df = pd.DataFrame(table_data, columns=headers)
    results_df[results_df.select_dtypes(include=[np.number]).columns] = \
        results_df.select_dtypes(include=[np.number]).round(4)
    
    results_df.to_csv("recommender_system_results.csv", index=False)
    print("\nResults saved to 'recommender_system_results.csv'")


if __name__ == "__main__":
    main()