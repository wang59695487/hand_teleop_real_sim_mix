import torch.nn as nn


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTLoss(nn.Module):

    def __init__(self, w_kl_loss):
        super().__init__()
        self.w_kl_loss = w_kl_loss
        self.kl_div_fn = kl_divergence
        self.act_loss_fn = nn.L1Loss(reduction="none")

    def forward(self, act_pred, act_true, is_pad, mu, log_var):
        kl_div, _, _ = self.kl_div_fn(mu, log_var)
        act_loss = self.act_loss_fn(act_pred, act_true)
        act_loss = (act_loss * ~is_pad.unsqueeze(dim=-1)).mean()
        total_loss = act_loss + kl_div[0] * self.w_kl_loss
        loss_dict = {
            "action_loss": act_loss,
            "kl_div": kl_div,
            "loss": total_loss
        }

        return loss_dict
