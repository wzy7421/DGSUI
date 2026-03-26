import torch
import torch.optim as optim
from models.dgsui import DGSUI
from utils.loss import orthogonal_loss, reconstruction_loss, shannon_entropy_loss, adaptive_margin_bpr_loss


def train_one_epoch(model, dataloader, optimizer, config):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        user_ids, seq_item_ids, time_intervals, pos_items, neg_items, pos_popularity = batch

        optimizer.zero_grad()

        # Forward pass
        final_user_emb, h_dyn, g, s, routing_weights_s = model(user_ids, seq_item_ids, time_intervals)

        # Predictions
        pos_scores = model.predict(final_user_emb, pos_items)
        neg_scores = model.predict(final_user_emb, neg_items)

        # --- Calculate Eq (10): Total Loss ---

        # 1. Main Recommendation Loss (Adaptive Margin BPR)
        l_main = adaptive_margin_bpr_loss(pos_scores, neg_scores, pos_popularity, gamma=config['gamma'])

        # 2. Disentanglement Loss (Orthogonal + Recon)
        l_ortho = orthogonal_loss(g, s)
        l_recon = reconstruction_loss(h_dyn, g, s)
        l_disentangle = l_ortho + config['lambda_r'] * l_recon

        # 3. Entropy Regularization Loss (Pareto improvement)
        # Note: We want to MAXIMIZE entropy, so we subtract it in the loss function
        l_entropy = shannon_entropy_loss(routing_weights_s)

        # Total Objective
        # L_total = L_main + alpha * L_disentangle - beta * L_entropy + L2_Reg
        loss = l_main + config['alpha'] * l_disentangle - config['beta'] * l_entropy

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    # Hyperparameters from paper Table S1 / Parameters section
    config = {
        'hidden_size': 64,
        'num_layers': 3,
        'gamma': 0.1,  # Adaptive margin scaler
        'alpha': 0.5,  # Disentanglement weight
        'beta': 0.05,  # Entropy weight
        'lambda_r': 0.1,  # Reconstruction weight
        'lr': 0.001
    }

    print("DGSUI Model Initialized. Ready for training...")
    # Initialize model, dataloader, and train