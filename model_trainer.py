from surprise import KNNBasic, SVD
from surprise.prediction_algorithms.knns import KNNWithMeans
from tqdm import tqdm

def train_models(trainset):
    ubcf = KNNBasic(sim_options={'user_based': True})
    ibcf = KNNWithMeans(sim_options={'user_based': False})
    svd = SVD()

    models = [ubcf, ibcf, svd]

    for model in tqdm(models, desc="Training models"):
        model.fit(trainset)

    return models