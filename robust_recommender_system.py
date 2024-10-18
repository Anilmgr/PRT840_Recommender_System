"""
Robust Recommender System Implementation
======================================

This module implements a robust recommender system with attack detection capabilities.
It supports multiple recommendation algorithms including User-Based CF, Item-Based CF, 
and SVD, along with evaluation under different attack scenarios.

Dependencies:
    - pandas
    - numpy
    - surprise
    - tabulate
    - logging

Author: [Your Name]
Date: October 2024
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, SVD, KNNWithMeans
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
from collections import defaultdict
from tabulate import tabulate
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataPreprocessor:
    """Handles data loading and preprocessing operations."""
    
    @staticmethod
    def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
        """
        Load and preprocess the ratings data from a file.
        
        Args:
            file_path: Path to the ratings data file
            
        Returns:
            DataFrame containing cleaned ratings data
            
        Raises:
            Exception: If data loading or preprocessing fails
        """
        try:
            ratings = pd.read_csv(
                file_path,
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"]
            )
            logging.info(f"Original dataset shape: {ratings.shape}")
            
            # Clean data
            ratings.drop_duplicates(inplace=True)
            ratings.dropna(inplace=True)
            
            # Filter users and items
            user_counts = ratings["user_id"].value_counts()
            item_counts = ratings["item_id"].value_counts()
            valid_users = user_counts[user_counts >= 5].index
            valid_items = item_counts[item_counts >= 3].index
            
            ratings = ratings[
                ratings["user_id"].isin(valid_users) & 
                ratings["item_id"].isin(valid_items)
            ]
            
            # Ensure ratings are within valid range
            ratings = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]
            ratings.reset_index(drop=True, inplace=True)
            
            logging.info(f"Dataset shape after cleaning: {ratings.shape}")
            return ratings
            
        except Exception as e:
            logging.error(f"Error in load_and_preprocess_data: {str(e)}")
            raise

class RobustRecommenderSystem:
    """Main recommender system implementation."""
    
    @staticmethod
    def prepare_surprise_data(ratings: pd.DataFrame) -> Dataset:
        """
        Convert pandas DataFrame to Surprise Dataset format.
        
        Args:
            ratings: DataFrame containing ratings data
            
        Returns:
            Surprise Dataset object
        """
        reader = Reader(rating_scale=(1, 5))
        return Dataset.load_from_df(
            ratings[["user_id", "item_id", "rating"]],
            reader
        )

    @staticmethod
    def tune_model(
        model_class: Any,
        param_grid: Dict,
        data: Dataset
    ) -> Any:
        """
        Perform grid search to find optimal model parameters.
        
        Args:
            model_class: Surprise algorithm class
            param_grid: Dictionary of parameters to search
            data: Training data
            
        Returns:
            Tuned model with best parameters
        """
        gs = GridSearchCV(
            model_class,
            param_grid,
            measures=["rmse", "mae"],
            cv=5
        )
        gs.fit(data)
        
        best_params = gs.best_params["rmse"]
        best_score = gs.best_score["rmse"]
        
        logging.info(f"Best parameters for {model_class.__name__}: {best_params}")
        logging.info(f"Best RMSE score: {best_score}")
        
        return gs.best_estimator["rmse"]

class AttackGenerator:
    """Generates attack profiles for robustness testing."""
    
    @staticmethod
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
            num_items: Total number of items
            target_item: Target item ID
            filler_size: Proportion of items to fill with ratings
            
        Returns:
            Array of attack profiles
        """
        attack_profiles = []
        for _ in range(num_fake_users):
            profile = np.random.randint(1, 6, size=int(num_items * filler_size))
            profile = np.pad(profile, (0, num_items - len(profile)), "constant")
            profile[target_item] = 5
            attack_profiles.append(profile)
        return np.array(attack_profiles)
    
    @staticmethod
    def generate_average_attack(
        num_fake_users: int,
        num_items: int,
        target_item: int,
        filler_size: float = 0.1,
        ratings_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generate average attack profiles based on item rating means.
        
        Args:
            num_fake_users: Number of fake users to generate
            num_items: Total number of items
            target_item: Target item ID
            filler_size: Proportion of items to fill with ratings
            ratings_data: DataFrame containing ratings
            
        Returns:
            Array of attack profiles
        """
        item_means = ratings_data.groupby("item_id")["rating"].mean()
        attack_profiles = []
        for _ in range(num_fake_users):
            profile = item_means.sample(n=int(num_items * filler_size)).round().values
            profile = np.pad(profile, (0, num_items - len(profile)), "constant")
            profile[target_item] = 5
            attack_profiles.append(profile)
        return np.array(attack_profiles)

class Evaluator:
    """Handles model evaluation and metrics calculation."""
    
    @staticmethod
    def dcg_score(relevance: List[int], k: int) -> float:
        """Calculate Discounted Cumulative Gain."""
        relevance = np.asarray(relevance)[:k]
        if relevance.size:
            return np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))
        return 0.0

    @staticmethod
    def ndcg_score(relevance: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        best = sorted(relevance, reverse=True)
        dcg = Evaluator.dcg_score(relevance, k)
        idcg = Evaluator.dcg_score(best, k)
        return dcg / idcg if idcg != 0 else 0.0

    @staticmethod
    def evaluate_model(
        model: Any,
        testset: List,
        attack_profiles: Optional[np.ndarray] = None,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate model performance with various metrics.
        
        Args:
            model: Trained recommender model
            testset: Test data
            attack_profiles: Optional attack profiles
            k: Number of recommendations to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if attack_profiles is not None:
            # Include attack profiles in training data
            original_ratings = [
                (model.trainset.to_raw_uid(u), model.trainset.to_raw_iid(i), r)
                for (u, i, r) in model.trainset.all_ratings()
            ]
            
            attack_ratings = [
                (f"fake_user_{i}", j, rating)
                for i, profile in enumerate(attack_profiles)
                for j, rating in enumerate(profile)
                if rating != 0
            ]
            
            all_ratings = original_ratings + attack_ratings
            df = pd.DataFrame(all_ratings, columns=["user", "item", "rating"])
            
            reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
            data = Dataset.load_from_df(df, reader)
            
            new_trainset = data.build_full_trainset()
            model.fit(new_trainset)
        
        predictions = model.test(testset)
        
        # Calculate basic metrics
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        # Calculate ranking metrics
        user_predictions = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_predictions[uid].append((est, true_r, iid))
        
        metrics = {
            "hit_count": 0,
            "arhr_sum": 0,
            "ap_sum": 0,
            "ndcg_sum": 0
        }
        
        for user_ratings in user_predictions.values():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            top_k = user_ratings[:k]
            
            relevance = [1 if true_r >= 4 else 0 for _, true_r, _ in top_k]
            
            if 1 in relevance:
                metrics["hit_count"] += 1
                rank = relevance.index(1) + 1
                metrics["arhr_sum"] += 1 / rank
            
            ap = sum(
                sum(relevance[: i + 1]) / (i + 1) if rel else 0
                for i, rel in enumerate(relevance)
            ) / k
            metrics["ap_sum"] += ap
            
            ndcg = Evaluator.ndcg_score(relevance, k)
            metrics["ndcg_sum"] += ndcg
        
        num_users = len(user_predictions)
        
        return {
            "RMSE": rmse,
            "MAE": mae,
            f"Hit Rate@{k}": metrics["hit_count"] / num_users,
            f"ARHR@{k}": metrics["arhr_sum"] / num_users,
            f"MAP@{k}": metrics["ap_sum"] / num_users,
            f"NDCG@{k}": metrics["ndcg_sum"] / num_users
        }

def main():
    """Main execution function."""
    try:
        # Load and preprocess data
        ratings = DataPreprocessor.load_and_preprocess_data("ml-100k/u.data")
        data = RobustRecommenderSystem.prepare_surprise_data(ratings)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Define parameter grids
        param_grids = {
            "ubcf": {
                "k": [20, 40, 60],
                "sim_options": {
                    "name": ["cosine", "msd", "pearson", "pearson_baseline"],
                    "user_based": [True]
                }
            },
            "ibcf": {
                "k": [20, 40, 60],
                "sim_options": {
                    "name": ["cosine", "msd", "pearson", "pearson_baseline"],
                    "user_based": [False]
                }
            },
            "svd": {
                "n_factors": [50, 100, 150],
                "n_epochs": [20, 30],
                "lr_all": [0.002, 0.005],
                "reg_all": [0.02, 0.1]
            }
        }
        
        # Train models
        models = {
            "UBCF": RobustRecommenderSystem.tune_model(KNNBasic, param_grids["ubcf"], data),
            "IBCF": RobustRecommenderSystem.tune_model(KNNWithMeans, param_grids["ibcf"], data),
            "SVD": RobustRecommenderSystem.tune_model(SVD, param_grids["svd"], data)
        }
        
        # Generate attack profiles
        num_items = ratings["item_id"].nunique()
        target_item = np.random.randint(0, num_items)
        num_fake_users = int(0.01 * ratings["user_id"].nunique())
        
        attack_scenarios = [
            ("No Attack", None),
            ("Random Attack", AttackGenerator.generate_random_attack(
                num_fake_users, num_items, target_item
            )),
            ("Average Attack", AttackGenerator.generate_average_attack(
                num_fake_users, num_items, target_item, ratings_data=ratings
            ))
        ]
        
        # Evaluate models
        results = {}
        for model_name, model in models.items():
            model.fit(trainset)
            results[model_name] = {
                scenario: Evaluator.evaluate_model(model, testset, attack_profiles)
                for scenario, attack_profiles in attack_scenarios
            }
        
        # Format and save results
        table_data = [
            [model_name, scenario] + list(metrics.values())
            for model_name, scenarios in results.items()
            for scenario, metrics in scenarios.items()
        ]
        
        headers = ["Model", "Scenario"] + list(
            next(iter(results.values()))["No Attack"].keys()
        )
        
        logging.info("\nResults:")
        logging.info(
            tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="grid")
        )
        
        # Save results to CSV
        results_df = pd.DataFrame(table_data, columns=headers)
        results_df[results_df.select_dtypes(include=[np.number]).columns] = \
            results_df.select_dtypes(include=[np.number]).round(4)
        results_df.to_csv("robust_recommender_system_results.csv", index=False)
        logging.info("Results saved to 'robust_recommender_system_results.csv'")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()