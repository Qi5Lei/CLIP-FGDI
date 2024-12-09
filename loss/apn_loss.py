import torch
import torch.nn.functional as F


def cosine_dist(x, y):
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos_sim(x,y)


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    dist1 = torch.pow(x - y, 2).sum(1)
    dist1 = dist1.clamp(min=1e-12).sqrt()

    return dist1






class APNLoss(torch.nn.Module):
    def __init__(self, epsilon=0.01, use_gpu=True):
        super(APNLoss, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu

    def forward(self, a, p, n):
        # [128, 512] a
        # [128, 512] p
        # [128, 512] n
        # B = len(ac)
        # pn = torch.cat((ap, an), dim=0)
        # logits = torch.div(torch.matmul(ac, pn.T), 1.)
        #
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits = logits - logits_max.detach()
        # exp_logits = torch.exp(logits)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask_p = torch.zeros((log_prob.size(0), log_prob.size(1))).to(ac.device)
        # mask_p[[i for i in range(B)], [i for i in range(B)]] = 1
        # mean_log_prob_pos = (mask_p * log_prob).sum(1) / mask_p.sum(1)
        # loss = - mean_log_prob_pos.mean()
        # return loss

        dist_ap = cosine_dist(a, p)
        dist_an = cosine_dist(a, n)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        return loss
