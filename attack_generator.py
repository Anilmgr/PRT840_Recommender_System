import numpy as np

def generate_random_attack(num_fake_users, num_items, target_item, filler_size=0.1):
    attack_profiles = []
    for _ in range(num_fake_users):
        profile = np.random.randint(1, 6, size=int(num_items * filler_size))
        profile = np.pad(profile, (0, num_items - len(profile)), 'constant')
        profile[target_item] = 5
        attack_profiles.append(profile)
    return np.array(attack_profiles)

def generate_average_attack(num_fake_users, num_items, target_item, filler_size=0.1, ratings_data=
                            {}):
    item_means = ratings_data.groupby('item_id')['rating'].mean()
    attack_profiles = []
    for _ in range(num_fake_users):
        profile = item_means.sample(n=int(num_items * filler_size)).round().values
        profile = np.pad(profile, (0, num_items - len(profile)), 'constant')
        profile[target_item] = 5
        attack_profiles.append(profile)
    return np.array(attack_profiles)

def generate_attacks(ratings):
    num_items = ratings['item_id'].nunique()
    target_item = np.random.randint(0, num_items)
    num_fake_users = int(0.01 * ratings['user_id'].nunique())

    print("Generating attack profiles...")
    random_attack = generate_random_attack(num_fake_users, num_items, target_item)
    average_attack = generate_average_attack(num_fake_users, num_items, target_item, ratings_data=ratings)
    print("Attack profiles generated.")

    return random_attack, average_attack