import torch.nn.functional as F
from .triplet_loss import TripletLoss
from .apn_loss import APNLoss
from .center_loss import CenterLoss
from .contrastive_loss import SupConLoss
from .softmax_loss import CrossEntropyLabelSmooth


def make_loss(num_classes):
    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    apnloss = APNLoss()

    def loss_func(score, feat, target, i2tscore=None, image_feature=None, text_p=None, text_n=None, beta=0.2):
        if isinstance(score, list):
            ID_LOSS = [xent(scor, target) for scor in score[0:]]
            ID_LOSS = sum(ID_LOSS)
        else:
            ID_LOSS = xent(score, target)

        if isinstance(feat, list):
            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
            TRI_LOSS = sum(TRI_LOSS)
        else:
            TRI_LOSS = triplet(feat, target)[0]
        loss = 1.0 * TRI_LOSS + 1.0 * ID_LOSS
        if i2tscore is not None:
            I2TLOSS = xent(i2tscore, target)
            loss = (1.0 - beta) * I2TLOSS + 1.0 * loss
        if text_p is not None and text_n is not None:
            APNloss = apnloss(image_feature, text_p, text_n)
            loss = beta * APNloss + 1.0 * loss
        return loss

    return loss_func
