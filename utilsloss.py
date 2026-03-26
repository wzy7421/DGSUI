import torch
import torch.nn.functional as F


def orthogonal_loss(g, s):
    """
    Eq (4): L_ortho = |g^T s| / (||g|| * ||s||)
    Enforces geometric orthogonality between habitual and exploratory intents.
    """
    g_norm = F.normalize(g, p=2, dim=-1)
    s_norm = F.normalize(s, p=2, dim=-1)
    cosine_sim = torch.sum(g_norm * s_norm, dim=-1)
    return torch.mean(torch.abs(cosine_sim))


def reconstruction_loss(h_dyn, g, s):
    """
    Eq (5): L_recon = || h - (g+s) ||^2
    """
    return torch.mean(torch.norm(h_dyn - (g + s), p=2, dim=-1) ** 2)


def shannon_entropy_loss(routing_weights):
    """
    Eq (7): L_entropy = - \sum p \log p
    Maximizes entropy to proactively seek out 'weak ties' / topological diversity.
    *Note: We return negative entropy because we want to maximize it (minimize negative).*
    """
    # Add small epsilon to prevent log(0)
    entropy = - torch.sum(routing_weights * torch.log(routing_weights + 1e-9), dim=-1)
    return torch.mean(entropy)


def adaptive_margin_bpr_loss(pos_scores, neg_scores, item_popularity, gamma=0.1):
    """
    Eq (8) & (9): Long-tail debiasing optimization via Adaptive Margin.
    m_i = \gamma * \log(1 + N_i^{-1})
    L_main = -\ln \sigma(y_pos - y_neg - m_i)
    """
    # Calculate margin based on popularity (N_i)
    # The less popular the item, the larger the margin, pushing it higher in rank
    margin = gamma * torch.log(1.0 + 1.0 / (item_popularity + 1.0))

    loss = - F.logsigmoid(pos_scores - neg_scores - margin)
    return torch.mean(loss)