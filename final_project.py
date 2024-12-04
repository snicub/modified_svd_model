import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pprint import pprint

#step 1: load data into dataframe
def load_data(file_path):
    df = pd.read_json(file_path, lines=True)
    required_columns = {'reviewerID', 'asin', 'overall'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    return df

def preprocess_data(df):
    user_ids = df['reviewerID'].astype('category').cat.codes
    item_ids = df['asin'].astype('category').cat.codes
    ratings = df['overall']
    return user_ids, item_ids, ratings

#step 2: define svd model itself
class SVDModel:
    def __init__(self, num_users, num_items, latent_factors=50, lr=0.01, reg=0.02):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_factors = latent_factors
        self.lr = lr
        self.reg = reg
        
        # create user and latent facotrs
        self.user_factors = np.random.normal(0, 0.1, (num_users, latent_factors))
        self.item_factors = np.random.normal(0, 0.1, (num_items, latent_factors))
    #we want to predict rating for user item pair
    def predict(self, user_id, item_id):
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])

    #train the model using gradient descent approach 
    def train(self, user_ids, item_ids, ratings, num_epochs=10):
        for epoch in range(num_epochs):
            total_loss = 0
            for u, i, r in zip(user_ids, item_ids, ratings):
                #evaluate prediction and error
                pred = self.predict(u, i)
                error = r - pred
                
                #update latency factors
                user_gradient = -2 * error * self.item_factors[i] + 2 * self.reg * self.user_factors[u]
                item_gradient = -2 * error * self.user_factors[u] + 2 * self.reg * self.item_factors[i]
                
                self.user_factors[u] -= self.lr * user_gradient
                self.item_factors[i] -= self.lr * item_gradient
                
                #calc loss
                total_loss += error ** 2

            avg_loss = total_loss / len(user_ids)
            print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

#evaluate preidciotns
def evaluate_predictions(model, test_data):
    user_ids, item_ids, true_ratings = test_data
    predicted_ratings = [model.predict(u, i) for u, i in zip(user_ids, item_ids)]
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    rmse = mean_squared_error(true_ratings, predicted_ratings, squared=False)
    print("\nEvaluation Results:")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    return mae, rmse

#create top N recs in a csv
def evaluate_top_n(model, test_data, train_data, top_n=10):
    user_ids_test, item_ids_test, ratings_test = test_data
    user_ids_train, item_ids_train, _ = train_data
    user_ids_unique = np.unique(user_ids_test)

    top_n_recommendations = {}
    precision_list, recall_list, ndcg_list = [], [], []

    for user_id in user_ids_unique:
        #predict ratings for all except htose in training data
        items = np.setdiff1d(np.arange(model.num_items), item_ids_train[user_ids_train == user_id])
        predictions = [model.predict(user_id, item_id) for item_id in items]
        
        #retrieve top n items
        top_items = items[np.argsort(predictions)[-top_n:][::-1]]
        top_n_recommendations[user_id] = top_items.tolist()

        #calc precision, recall and NDCG
        relevant_items = item_ids_test[user_ids_test == user_id]
        hits = len(set(top_items) & set(relevant_items))
        precision = hits / top_n
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)

        #NDCG clauclation 
        dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(top_items) if item in relevant_items])
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(relevant_items), top_n))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f_measure = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0
    avg_ndcg = np.mean(ndcg_list)

    print("\nTop-N Recommendation Results:")
    print(f"  - Precision: {avg_precision:.4f}")
    print(f"  - Recall: {avg_recall:.4f}")
    print(f"  - F-Measure: {avg_f_measure:.4f}")
    print(f"  - NDCG: {avg_ndcg:.4f}\n")
    return avg_precision, avg_recall, avg_f_measure, avg_ndcg, top_n_recommendations

#determine the fairness
def evaluate_fairness(user_groups, recommendations):
    fairness_scores = {group: 0 for group in set(user_groups.values())}
    total_recommendations = 0

    for user_id, rec_items in recommendations.items():
        group = user_groups.get(user_id, "Unknown")
        fairness_scores[group] += len(rec_items)
        total_recommendations += len(rec_items)

    #normalize scores to compute percentage of recommendations by group
    fairness_scores = {group: round(count / total_recommendations, 4) for group, count in fairness_scores.items()}
    print("\nFairness Evaluation:")
    for group, score in fairness_scores.items():
        print(f"  {group}: {score * 100:.2f}% of recommendations")
    return fairness_scores

#determine transparency
def explain_recommendations(user_id, model, item_ids):
    explanations = {}
    user_vector = model.user_factors[user_id]
    for item_id in item_ids:
        item_vector = model.item_factors[item_id]
        contributions = user_vector * item_vector
        explanations[item_id] = {
            f"Feature {i+1}": round(contribution, 4) for i, contribution in enumerate(contributions)
        }
        explanations[item_id]['Total'] = round(sum(contributions), 4)
    pprint(explanations)

#determine controollablity 
def create_user_clusters(user_factors, num_clusters=3):
    scaler = StandardScaler()
    user_factors_scaled = scaler.fit_transform(user_factors)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(user_factors_scaled)
    print("\nUser Clusters:", clusters)
    return clusters

#privacy function using hashing 
def anonymize_user_ids(recommendations):
    return {hash(uid): items for uid, items in recommendations.items()}

#calc robustness
def simulate_attack(model, user_id, item_id, factor=10):
    original = model.predict(user_id, item_id)
    model.user_factors[user_id] *= factor
    model.item_factors[item_id] *= factor
    manipulated = model.predict(user_id, item_id)
    print(f"Original: {original:.4f}, Manipulated: {manipulated:.4f}")

#driver code 
def main_with_optional_tasks():
    #load data set
    df = load_data("Magazine_Subscriptions_5.json")
    user_ids, item_ids, ratings = preprocess_data(df)

    #train and split data
    train_idx, test_idx = train_test_split(np.arange(len(user_ids)), test_size=0.2, random_state=42)
    train_data = (user_ids[train_idx], item_ids[train_idx], ratings.iloc[train_idx].values)
    test_data = (user_ids[test_idx], item_ids[test_idx], ratings.iloc[test_idx].values)

    #initialize svd
    model = SVDModel(len(np.unique(user_ids)), len(np.unique(item_ids)))
    model.train(*train_data, num_epochs=10)

    #run evals 
    evaluate_predictions(model, test_data)
    _, _, _, _, recommendations = evaluate_top_n(model, test_data, train_data)

    #optional tasks using helper functions 
    explain_recommendations(0, model, [0, 1, 2])
    evaluate_fairness({uid: f"Group {uid % 3}" for uid in range(len(np.unique(user_ids)))}, recommendations)
    create_user_clusters(model.user_factors)
    anonymize_user_ids(recommendations)
    simulate_attack(model, 0, 1)

if __name__ == "__main__":
    main_with_optional_tasks()
