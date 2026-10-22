"""TabSyn's VAE loss, with FLIP's uniform attribute loss L_S (Eq. 3).
Only compute_loss is used by this repository; TabSyn's standalone training
script was removed."""
import torch
import torch.nn as nn

from tabsyn.vae.flip_fairness import (uniform_attribute_loss,
                                      uniform_attribute_loss_per_sample)


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z,
                 s_idx=None, Recon_S_logits=None, ls_mode='per_sample',
                 recon_s=False):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            # FLIP: by default exclude the protected attribute from the
            # reconstruction loss - its assignment should be random
            # (Sec. 4.2), and its logits are instead supervised by the
            # uniform attribute loss. recon_s=True keeps the paper-literal
            # setup where S stays reconstructable (needed for BER to
            # discriminate between lambda values).
            if s_idx is not None and idx == s_idx and not recon_s:
                continue
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

    # FLIP uniform attribute loss L_S (Eq. 3). ls_mode selects the variant:
    # 'per_sample' (default) is DP-SGD-decomposable so per-sample clipping
    # bounds each record's influence; 'batch_mean' is the paper-literal Eq. 3,
    # which only constrains the batch-average prediction and leaves room for
    # feature-dependent per-row predictions (what makes BER < 0.5 possible).
    loss_s = torch.tensor(0.0, device=X_num.device)
    if s_idx is not None and Recon_S_logits is not None:
        if ls_mode == 'batch_mean':
            loss_s = uniform_attribute_loss(Recon_S_logits)
        else:
            loss_s = uniform_attribute_loss_per_sample(Recon_S_logits)

    return mse_loss, ce_loss, loss_kld, acc, loss_s
