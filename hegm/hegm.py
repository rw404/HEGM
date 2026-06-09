import torch


def decode_predictions(preds: torch.Tensor):
    K = (preds.shape[-1] - 2) // 3
    viewtime_exp = preds[..., 0:1]
    viewtime_mu = preds[..., 1 : K + 1]
    weights_logits = preds[..., 2 * K + 1 :]

    lambda_score = torch.nn.functional.softplus(viewtime_exp, beta=0.5) + 1e-6
    exp_mean = 1.0 / lambda_score

    # Skip-watch decomposition
    prob_skip = torch.sigmoid(weights_logits[..., 0:1])
    weights_gmm = torch.nn.functional.softmax(weights_logits[..., 1:], dim=-1)
    weights = torch.cat([prob_skip, (1.0 - prob_skip) * weights_gmm], dim=-1)
    mu = torch.nn.functional.softplus(viewtime_mu)

    expected_viewtime = (torch.cat([exp_mean, mu], dim=-1) * weights).sum(dim=-1)
    return expected_viewtime


def hegm_loss(scores: torch.Tensor, y: torch.Tensor):
    K = (scores.shape[-1] - 2) // 3
    viewtime_exp = scores[..., 0:1]
    viewtime_mu = scores[..., 1 : K + 1]
    viewtime_sigma = scores[..., K + 1 : 2 * K + 1]
    weights_logits = scores[..., 2 * K + 1 :]

    # Skip-watch decomposition
    prob_skip = torch.sigmoid(weights_logits[..., 0:1])
    weights_gmm = torch.nn.functional.softmax(weights_logits[..., 1:], dim=-1)
    weights = torch.cat([prob_skip, (1.0 - prob_skip) * weights_gmm], dim=-1)

    lambda_score = torch.nn.functional.softplus(viewtime_exp, beta=0.5) + 1e-6
    exp_mean = 1.0 / lambda_score
    mu = torch.nn.functional.softplus(viewtime_mu)
    sigma = torch.nn.functional.softplus(viewtime_sigma) + 1e-6
    dispersion = sigma**2

    expected_viewtime = (torch.cat([exp_mean, mu], dim=-1) * weights).sum(dim=-1)
    mae_loss = torch.abs(expected_viewtime - y).mean()

    y_uns = y.unsqueeze(-1)
    # Exponential component for skip/short viewtimes
    log_exp = torch.log(lambda_score) - lambda_score * y_uns + torch.log(weights[..., 0:1] + 1e-6)
    # Gaussian mixture for engaged watching
    log_norm = (
        -0.5 * torch.log(2 * torch.pi * dispersion)
        - 0.5 * ((y_uns - mu) ** 2) / dispersion
        + torch.log(weights[..., 1:] + 1e-6)
    )

    loglikelihood = torch.logsumexp(torch.cat([log_exp, log_norm], dim=-1), dim=1)
    nll_loss = -torch.mean(loglikelihood)

    return nll_loss + mae_loss, nll_loss, mae_loss, expected_viewtime
