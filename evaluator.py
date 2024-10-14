import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from surprise import accuracy
from surprise import Dataset, Reader

def dcg_score(relevance, k):
    relevance = np.asarray(relevance)[:k]
    if relevance.size:
        return np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))
    return 0.0

def ndcg_score(relevance, k):
    best = sorted(relevance, reverse=True)
    dcg = dcg_score(relevance, k)
    idcg = dcg_score(best, k)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(model, testset, attack_profiles=None, k=10):
    if attack_profiles is not None:
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
        
        new_trainset = data.build_full_trainset()
        model.fit(new_trainset)
    
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
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
        
        relevance = [1 if true_r >= 4 else 0 for _, true_r, _ in top_k]
        
        if 1 in relevance:
            hit_count += 1
            rank = relevance.index(1) + 1
            arhr_sum += 1 / rank
        
        ap = sum([sum(relevance[:i+1]) / (i+1) if rel else 0 for i, rel in enumerate(relevance)]) / k
        ap_sum += ap
        
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

def evaluate_all_models(models, testset, random_attack, average_attack):
    attack_scenarios = [
        ('No Attack', None),
        ('Random Attack', random_attack),
        ('Average Attack', average_attack)
    ]

    results = {}
    for model in models:
        model_results = {}
        for scenario, attack_profiles in attack_scenarios:
            print(f"\nEvaluating {model.__class__.__name__} - {scenario}")
            model_results[scenario] = evaluate_model(model, testset, attack_profiles)
        results[model.__class__.__name__] = model_results

    return results