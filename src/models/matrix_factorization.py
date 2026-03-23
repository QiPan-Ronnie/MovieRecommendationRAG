"""
BPR Matrix Factorization using PyTorch.
Supports validation-based early stopping and hyperparameter search.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from src.evaluation.metrics import evaluate_all, print_results


class BPRDataset(Dataset):
    """Dataset for BPR training: (user, pos_item, neg_item) triples."""

    def __init__(self, train_df, all_movie_ids, neg_per_pos=1, seed=42):
        self.rng = np.random.RandomState(seed)
        self.all_movies = list(all_movie_ids)

        self.user_pos = defaultdict(set)
        for _, row in train_df.iterrows():
            self.user_pos[row["user_id"]].add(row["movie_id"])

        self.triples = []
        for _, row in train_df.iterrows():
            uid = row["user_id"]
            pos = row["movie_id"]
            for _ in range(neg_per_pos):
                neg = self.rng.choice(self.all_movies)
                while neg in self.user_pos[uid]:
                    neg = self.rng.choice(self.all_movies)
                self.triples.append((uid, pos, neg))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        u, p, n = self.triples[idx]
        return u, p, n


class MFModel(nn.Module):
    """Matrix Factorization with BPR loss."""

    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, user_ids, pos_ids, neg_ids):
        u = self.user_embed(user_ids)
        p = self.item_embed(pos_ids)
        n = self.item_embed(neg_ids)
        pos_score = (u * p).sum(dim=1)
        neg_score = (u * n).sum(dim=1)
        return pos_score, neg_score

    def predict(self, user_ids):
        """Predict scores for all items for given users."""
        u = self.user_embed(user_ids)
        all_items = self.item_embed.weight
        scores = torch.matmul(u, all_items.T)
        return scores


def bpr_loss(pos_score, neg_score):
    """BPR loss: -log(sigmoid(pos - neg))."""
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()


def evaluate_on_split(model, split_df, train_user_items, user2idx, movie2idx,
                      idx2movie, device, k=10):
    """Evaluate model on a given split (validation or test)."""
    ground_truth = defaultdict(set)
    for _, row in split_df.iterrows():
        ground_truth[row["user_id"]].add(row["movie_id"])

    model.eval()
    predictions = {}
    split_users = [u for u in ground_truth.keys() if u in user2idx]

    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(split_users), batch_size):
            batch_users = split_users[start:start + batch_size]
            user_indices = torch.tensor([user2idx[u] for u in batch_users]).to(device)
            scores = model.predict(user_indices).cpu().numpy()

            for i, uid in enumerate(batch_users):
                user_scores = scores[i]
                for mid in train_user_items[uid]:
                    if mid in movie2idx:
                        user_scores[movie2idx[mid]] = -float("inf")

                top_indices = np.argsort(user_scores)[::-1][:100]
                predictions[uid] = [idx2movie[idx] for idx in top_indices]

    total_items = len(movie2idx)
    results, per_user_ndcg = evaluate_all(predictions, ground_truth, k=k, total_items=total_items, ks=[1, 5, 10])
    return results, per_user_ndcg, predictions


def train_mf(train_df, embed_dim=64, lr=1e-3, reg=1e-5, epochs=50,
             batch_size=1024, patience=5, device="cpu",
             val_df=None):
    """Train BPR-MF model with optional validation-based early stopping."""
    all_users = sorted(train_df["user_id"].unique())
    all_movies = sorted(train_df["movie_id"].unique())
    user2idx = {u: i for i, u in enumerate(all_users)}
    movie2idx = {m: i for i, m in enumerate(all_movies)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    train_indexed = train_df.copy()
    train_indexed["user_id"] = train_indexed["user_id"].map(user2idx)
    train_indexed["movie_id"] = train_indexed["movie_id"].map(movie2idx)

    dataset = BPRDataset(train_indexed, list(range(len(all_movies))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MFModel(len(all_users), len(all_movies), embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    # Build train user items for masking during eval
    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[row["user_id"]].add(row["movie_id"])

    best_val_ndcg = -1
    best_loss = float("inf")
    val_no_improve = 0
    loss_no_improve = 0
    best_state = None
    use_val = val_df is not None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for users, pos_items, neg_items in loader:
            users = users.long().to(device)
            pos_items = pos_items.long().to(device)
            neg_items = neg_items.long().to(device)

            pos_score, neg_score = model(users, pos_items, neg_items)
            loss = bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if use_val and (epoch + 1) % 5 == 0:
            # Validation-based early stopping (checked every 5 epochs)
            val_results, _, _ = evaluate_on_split(
                model, val_df, train_user_items, user2idx, movie2idx,
                idx2movie, device, k=10
            )
            val_ndcg = val_results["NDCG@10"]
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val NDCG@10: {val_ndcg:.4f}")
            if val_ndcg > best_val_ndcg + 1e-4:
                best_val_ndcg = val_ndcg
                val_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                val_no_improve += 1
                if val_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (val NDCG)")
                    break
        elif not use_val:
            # Loss-based early stopping (only when no validation set)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                loss_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                loss_no_improve += 1
                if loss_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (train loss)")
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, user2idx, movie2idx, idx2movie


def run_mf(train_path="data/processed/train.csv",
           val_path="data/processed/val.csv",
           test_path="data/processed/test.csv",
           output_path="results/mf_scores.csv",
           embed_dim=64, lr=1e-3, epochs=50, k=10):
    """Run full MF pipeline with validation-based model selection."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Training MF (dim={embed_dim}, lr={lr})...")
    model, user2idx, movie2idx, idx2movie = train_mf(
        train_df, embed_dim=embed_dim, lr=lr, epochs=epochs,
        device=device, val_df=val_df
    )

    # Build train user items
    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[row["user_id"]].add(row["movie_id"])

    # Evaluate on test set
    print("Evaluating on test set...")
    results, per_user_ndcg, predictions = evaluate_on_split(
        model, test_df, train_user_items, user2idx, movie2idx,
        idx2movie, device, k=k
    )

    # Save scores for ranker
    all_scores = []
    for uid, ranked in predictions.items():
        user_idx = torch.tensor([user2idx[uid]]).to(device)
        with torch.no_grad():
            scores = model.predict(user_idx).cpu().numpy()[0]
        for movie_id in ranked:
            if movie_id in movie2idx:
                all_scores.append({
                    "user_id": uid,
                    "movie_id": movie_id,
                    "mf_score": float(scores[movie2idx[movie_id]])
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(all_scores).to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

    print_results(results, f"MF (dim={embed_dim})")
    return results, per_user_ndcg


if __name__ == "__main__":
    run_mf()
