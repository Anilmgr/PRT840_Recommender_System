import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate

# Load the MovieLens 100K dataset
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

print(f"Dataset shape: {ratings.shape}")
print(f"\nRatings distribution:\n{ratings['rating'].value_counts().sort_index()}")

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define collaborative filtering models
ubcf = KNNBasic(sim_options={'user_based': True})
ibcf = KNNWithMeans(sim_options={'user_based': False})
svd = SVD()

models = [ubcf, ibcf, svd]

# Train the models
for model in tqdm(models, desc="Training models"):
    model.fit(trainset)

def generate_random_attack(num_fake_users, num_items, target_item, filler_size=0.1):
    """
    Generate random attack profiles.
    
    :param num_fake_users: Number of fake users to generate
    :param num_items: Total number of items in the dataset
    :param target_item: The item to be pushed
    :param filler_size: Proportion of items to be rated by each fake user
    :return: Array of attack profiles
    """
    attack_profiles = []
    for _ in range(num_fake_users):
        profile = np.random.randint(1, 6, size=int(num_items * filler_size))
        profile = np.pad(profile, (0, num_items - len(profile)), 'constant')
        profile[target_item] = 5  # Set target item rating to maximum
        attack_profiles.append(profile)
    return np.array(attack_profiles)

def generate_average_attack(num_fake_users, num_items, target_item, filler_size=0.1, ratings_data=ratings):
    """
    Generate average attack profiles.
    
    :param num_fake_users: Number of fake users to generate
    :param num_items: Total number of items in the dataset
    :param target_item: The item to be pushed
    :param filler_size: Proportion of items to be rated by each fake user
    :param ratings_data: The original ratings data
    :return: Array of attack profiles
    """
    item_means = ratings_data.groupby('item_id')['rating'].mean()
    attack_profiles = []
    for _ in range(num_fake_users):
        profile = item_means.sample(n=int(num_items * filler_size)).round().values
        profile = np.pad(profile, (0, num_items - len(profile)), 'constant')
        profile[target_item] = 5  # Set target item rating to maximum
        attack_profiles.append(profile)
    return np.array(attack_profiles)

# Generate attack profiles
num_items = ratings['item_id'].nunique()
target_item = np.random.randint(0, num_items)
num_fake_users = int(0.01 * ratings['user_id'].nunique())  # 1% of total users

print("Generating attack profiles...")
random_attack_profiles = generate_random_attack(num_fake_users, num_items, target_item)
average_attack_profiles = generate_average_attack(num_fake_users, num_items, target_item, ratings_data=ratings)
print("Attack profiles generated.")

def dcg_score(relevance, k):
    """
    Calculate Discounted Cumulative Gain (DCG) at rank k.
    
    :param relevance: List of relevance scores
    :param k: Rank to calculate DCG at
    :return: DCG score
    """
    relevance = np.asarray(relevance)[:k]
    if relevance.size:
        return np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))
    return 0.0

def ndcg_score(relevance, k):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.
    
    :param relevance: List of relevance scores
    :param k: Rank to calculate NDCG at
    :return: NDCG score
    """
    best = sorted(relevance, reverse=True)
    dcg = dcg_score(relevance, k)
    idcg = dcg_score(best, k)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(model, testset, attack_profiles=None, k=10):
    """
    Evaluate a model's performance with various metrics.
    
    :param model: The recommender model to evaluate
    :param testset: The test set to evaluate on
    :param attack_profiles: Attack profiles to inject (if any)
    :param k: The k value for top-k metrics
    :return: Dictionary of evaluation metrics
    """
    if attack_profiles is not None:
        # Create a new dataset that includes both original ratings and attack profiles
        original_ratings = [(model.trainset.to_raw_uid(u), model.trainset.to_raw_iid(i), r) 
                            for (u, i, r) in model.trainset.all_ratings()]
        
        attack_ratings = []
        for i, profile in enumerate(attack_profiles):
            for j, rating in enumerate(profile):
                if rating != 0:
                    attack_ratings.append((f'fake_user_{i}', j, rating))
        
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
    
    # Calculate hit rate, ARHR, MAP@K, and NDCG@K
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r, iid))
    
    hit_count = 0
    arhr_sum = 0
    ap_sum = 0
    ndcg_sum = 0
    
    for uid, user_ratings in tqdm(user_est_true.items(), desc="Evaluating users", leave=False):
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        
        # Binary relevance: consider items with true rating >= 4 as relevant
        relevance = [1 if true_r >= 4 else 0 for _, true_r, _ in top_k]
        
        # Hit Rate and ARHR
        if 1 in relevance:
            hit_count += 1
            rank = relevance.index(1) + 1
            arhr_sum += 1 / rank
        
        # MAP@K
        ap = sum([sum(relevance[:i+1]) / (i+1) if rel else 0 for i, rel in enumerate(relevance)]) / k
        ap_sum += ap
        
        # NDCG@K
        ndcg = ndcg_score(relevance, k)
        ndcg_sum += ndcg
    
    num_users = len(user_est_true)
    hit_rate = hit_count / num_users
    arhr = arhr_sum / num_users
    map_k = ap_sum / num_users
    ndcg_k = ndcg_sum / num_users
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        f'Hit Rate@{k}': hit_rate,
        f'ARHR@{k}': arhr,
        f'MAP@{k}': map_k,
        f'NDCG@{k}': ndcg_k
    }

# Evaluate models before and after attacks
attack_scenarios = [
    ('No Attack', None),
    ('Random Attack', random_attack_profiles),
    ('Average Attack', average_attack_profiles)
]

results = {}

for model in models:
    model_results = {}
    for scenario, attack_profiles in attack_scenarios:
        print(f"\nEvaluating {model.__class__.__name__} - {scenario}")
        model_results[scenario] = evaluate_model(model, testset, attack_profiles)
    results[model.__class__.__name__] = model_results

# Prepare data for tabulation
table_data = []
headers = ["Model", "Scenario"] + list(next(iter(results.values()))["No Attack"].keys())

for model_name, scenarios in results.items():
    for scenario, metrics in scenarios.items():
        row = [model_name, scenario] + list(metrics.values())
        table_data.append(row)

# Print results in a table
print("\nResults:")
print(tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="grid"))

# Save the results to a CSV file
results_df = pd.DataFrame(table_data, columns=headers)

# Round numeric columns to 4 decimal places
numeric_columns = results_df.select_dtypes(include=[np.number]).columns
results_df[numeric_columns] = results_df[numeric_columns].round(4)

results_df.to_csv("recommender_system_results.csv", index=False)
print("\nResults saved to 'recommender_system_results.csv'")