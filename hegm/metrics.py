import torch


@torch.no_grad()
def batch_roc_auc_consumptions(targets: torch.Tensor, preds: torch.Tensor):
    n = targets.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=targets.device), torch.tensor(0.0, device=targets.device)

    dscores = targets.unsqueeze(1) - targets.unsqueeze(0)
    dpreds = preds.unsqueeze(1) - preds.unsqueeze(0)

    mask_upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=targets.device), diagonal=1)
    valid_mask = mask_upper & (dscores != 0.0)

    corrects = torch.sum(valid_mask & (dpreds != 0.0) & ((dpreds > 0) == (dscores > 0)))
    ties = torch.sum(valid_mask & (dpreds == 0.0))
    n_combinations = torch.sum(valid_mask)

    return (corrects + 0.5 * ties).float(), n_combinations.float()
