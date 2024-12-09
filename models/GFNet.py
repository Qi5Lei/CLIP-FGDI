import math
import random
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=16, w=7):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=(16, 12),i=0):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)

        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)

        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class GFNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0, is_domain_embedding=True, is_token_mix=True):

        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(patch_size)
        self.height = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.width = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        self.IN_ = nn.InstanceNorm2d(3, affine=True)
        print('using linear droppath with expect rate', drop_path_rate * 0.5)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.classifier = nn.Linear(384, 4)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=16, w=7)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.grl = GRL()
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, domain):
        x = self.IN_(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x).mean(1)
        return x

    def forward_eval(self, x):
        x = self.IN_(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x).mean(1)
        return x

    def forward(self, x, domain=None):
        if self.training:
            x = self.forward_features(x, domain)
            cls_score = self.classifier(self.grl(x))
            return x,cls_score
        else:
            x = self.forward_eval(x)
            return x

    def load_param(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        checkpoint = checkpoint_filter_fn(checkpoint, self)
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'aux_head.weight', 'aux_head.bias']:
            if k in checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        a, b = self.load_state_dict(checkpoint, strict=False)
        print(a)
        print(b)


def resize_pos_embed(posemb, posemb_newhm, height, width):
    posemb_token, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_newhm.shape,
                                                                                                height, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def resize_filiter(filter, target):
    filter = filter.permute(3, 2, 1, 0)
    height, width = target.shape[1], target.shape[0]
    res = F.interpolate(filter, size=(height, width), mode='bilinear')
    res = res.permute(3, 2, 1, 0)

    return res


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed, model.height, model.width)
        elif 'filter' in k:
            v = resize_filiter(v, model.state_dict()[k])
        out_dict[k] = v
    return out_dict



class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()
    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

class GRL(torch.nn.Module):
    def __init__(self, lambd=1.):
        super(GRL,self).__init__()
        self.lambd = lambd
    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x,lam)


if __name__ == "__main__":
    x = torch.zeros((128, 3, 384, 128))
    Gmodel = GFNet(
        img_size=(384, 128),
        patch_size=16, embed_dim=384, depth=19, mlp_ratio=4, drop_path_rate=0.15,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    print(Gmodel(x).shape)
