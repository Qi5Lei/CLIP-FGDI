import os
import time
import argparse
import warnings
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings("ignore")
from reidutils.meter import AverageMeter
from reidutils.metrics import R1_mAP_eval
from cfgs import *
from reidutils import setup_logger
import datetime
from models import *
from functools import partial
from torch.cuda import amp
from torch import nn
from datasets.build import build_data_loader
from loss import make_loss
from solver import create_scheduler, WarmupMultiStepLR, make_optimizer_for_IE, \
    make_optimizer_prompt_domain, make_optimizer_prompt
import sys
from loss.supcontrast import SupConLoss

sys.path.append('/')


def get_model(args):
    if args.model == 'ViT':
        model = ViT(img_size=args.size_train,
                    stride_size=16,
                    drop_path_rate=0.1,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    qkv_bias=True)
        model.load_param(args.pretrain_vit_path)
    elif args.model == 'gfnet':
        model = GFNet(
            img_size=(384, 128),
            patch_size=16, embed_dim=384, depth=19, mlp_ratio=4, drop_path_rate=0.15,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        model.load_param(args.pretrain_gfnet_path)
    elif args.model == 'clip':
        model = Clip(sum(args.classes), args,domain_num=len(args.train_datasets),epsilon=args.epsilon)
    else:
        model = ViT(img_size=args.size_train,
                    stride_size=16,
                    drop_path_rate=0.1,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    qkv_bias=True)
        model.load_param(args.pretrain_path)
    return model


def train_stage_prior(train_loader_stage2, model, criterion, optimizer, scheduler, args_train,
                      logger_train, log_path, epochs=3):
    logger_train.info('start stage-prior training')
    device = 'cuda'
    loss_meter = AverageMeter()
    scaler = amp.GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter.reset()
        scheduler.step()
        for n_iter, (img, vid, _, _, domain, _) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = None
            target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam,
                                                    view_label=target_view)
                loss = criterion(score, feat, target, None)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            torch.cuda.synchronize()
            if (n_iter + 1) % args_train.log_period == 0:
                logger_train.info("STAGE2-Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                  .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                          loss_meter.avg, scheduler.get_lr()[0]))

        if epoch % args_train.checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(log_path,
                                    args_train.model + str(datetime.datetime.now()) + '_stageprior_{}.pth'.format(
                                        epoch)))


def train_stage1(train_loader_stage1, model, optimizer, scheduler, args_train, logger_train, log_path, get_domain=False,
                 epochs=120, omega=0.01):
    logger_train.info('start stage-1 training')
    device = 'cuda'
    loss_meter = AverageMeter()
    accd_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss('cuda')
    dc = nn.CrossEntropyLoss()
    image_features = []
    labels = []
    domains = []
    camids = []
    cids = []
    model.eval()
    with torch.no_grad():
        for n_iter, (img, pids, camid, viewids, domain, cid) in enumerate(train_loader_stage1):
            img = img.to(device)
            cid = cid.to(device)
            target = pids.to(device)
            camid = camid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)
                for i, domain, camid, img_feat, cid in zip(target, domain, camid, image_feature, cid):
                    labels.append(i)
                    domains.append(domain)
                    cids.append(cid)
                    camids.append(camid)
                    image_features.append(img_feat.cpu())

        labels_list = torch.stack(labels, dim=0).cuda()  # N
        domains_list = torch.stack(domains, dim=0).cuda()  # N
        cids_list = torch.stack(cids, dim=0).cuda()  # N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = args_train.batch_size
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
        del labels, image_features
        torch.cuda.empty_cache()

    for epoch in range(1, epochs + 1):
        model.train(True)
        loss_meter.reset()
        accd_meter.reset()
        scheduler.step(epoch)
        iter_list = torch.arange(num_image).to(device)
        for i in range(i_ter):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]
            target = labels_list[b_list]
            domain = domains_list[b_list]
            cid = cids_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                text_features, dcscore = model(label=target, get_text=True, domain=domain, cam_label=cid,
                                               getdomain=get_domain)
                text_features.requires_grad_(True)
                image_features.requires_grad_(True)
            loss_i2t = xent(image_features.float(), text_features.float(), target, target)
            loss_t2i = xent(text_features.float(), image_features.float(), target, target)

            dc_loss = dc(dcscore, domain)
            loss = loss_i2t + loss_t2i + omega * dc_loss

            accd = (dcscore.max(1)[1] == domain).float().mean()
            accd_meter.update(accd, 1)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % args_train.log_period == 0:
                logger_train.info(
                    "STAGE1-Epoch[{}] Iteration[{}/{}] Domain_Acc: {:.3f} Loss: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (i + 1), len(train_loader_stage1), accd_meter.avg,
                            loss_meter.avg, scheduler._get_lr(epoch)[0]))

    torch.save(model.promptlearner.state_dict(),
               os.path.join(log_path,
                            args_train.model + '_clsctx_' + str(get_domain) + '.pth'))


def train_stage2(train_loader_stage2, model, criterion, optimizer, scheduler, testloaders, args_train, logger_train,
                 logger_test, log_path, epochs=60):
    logger_train.info('start stage-2 training')
    device = 'cuda'
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    accd_meter = AverageMeter()
    accc_meter = AverageMeter()
    num_classes = sum(args_train.classes)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    dc = nn.CrossEntropyLoss()
    batch = args_train.batch_size
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    model.eval()
    p2d = train_loader_stage2.dataset.p2d
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature, _ = model(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()
    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        accd_meter.reset()
        accc_meter.reset()
        scheduler.step()
        for n_iter, (img, vid, _, _, domain, cid) in enumerate(train_loader_stage2):
            optimizer.zero_grad()

            img = img.to(device)
            target = vid.to(device)
            cid = cid.to(device)
            domain = domain.to(device)
            target_cam = None
            target_view = None
            model.eval()
            with torch.no_grad():
                text_feature_p, _ = model(label=target, get_text=True, domain=domain, getdomain=False)
                text_feature_n, _ = model(label=target, get_text=True, domain=domain, getdomain=True)
            model.train()
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam,
                                                    view_label=target_view)
                logits = image_features @ text_features.t()
                loss = criterion(score, feat, target, logits, image_features, text_feature_p.clone().detach(),
                                 text_feature_n.clone().detach(),beta=args_train.beta)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            acc = ((score[0] + score[1]).max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % args_train.log_period == 0:
                logger_train.info(
                    "STAGE2-Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, AccD: {:.3f}, AccC: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader_stage2),
                            loss_meter.avg, accd_meter.avg, accc_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        if epoch % args_train.checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(log_path,
                                    args_train.model + str(datetime.datetime.now()) + '_stage2_{}.pth'.format(epoch)))

        if epoch % args_train.eval_period == 0:
            test(testloaders, model, logger_test)


def test(testloaders, model, logger_test):
    model.eval()
    maps, r1s, r5s, r10s = [], [], [], []
    for name, val_loader in testloaders.items():
        evaluator = R1_mAP_eval(val_loader[1], max_rank=10, feat_norm=False, reranking=False)
        evaluator.reset()
        logger_test.info("Validation Results of {}: ".format(name))
        for n_iter, (img, pids, camids, viewids, domain, cid) in enumerate(val_loader[0]):
            with torch.no_grad():
                img = img.cuda()
                feat = model(img)
                evaluator.update((feat, pids, camids))
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger_test.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger_test.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger_test.info("-" * 30)
        torch.cuda.empty_cache()
        maps.append(mAP)
        r1s.append(cmc[0])
        r5s.append(cmc[4])
        r10s.append(cmc[9])
    logger_test.info("Average Results :")
    logger_test.info("Average, mAP:{:.1%}".format(sum(maps) / len(maps)))
    logger_test.info("Average, Rank-1:{:.1%}".format(sum(r1s) / len(r1s)))
    logger_test.info("Average, Rank-5:{:.1%}".format(sum(r5s) / len(r5s)))
    logger_test.info("Average, Rank-10:{:.1%}".format(sum(r10s) / len(r10s)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser_test = argparse.ArgumentParser(description='test')
    parsertrain, parsertest, logname = protocol_1(parser, parser_test)

    args_train = parsertrain.parse_args()
    args_test = parsertest.parse_args()
    time_now = str(datetime.datetime.now())[:-7]
    log_path = os.path.join(args_train.log_path, logname + '_' + args_train.backbone + '_' + time_now)
    logger_train = setup_logger(args_train.model + '_' + args_train.backbone + '_train', log_path, if_train=True)
    logger_test = setup_logger(args_train.model + '_' + args_train.backbone + '_test', log_path, if_train=False)
    logger_train.info("Log saved in- {}".format(log_path))
    logger_train.info("Training cfgs- {}".format(str(args_train)))
    logger_train.info("Running protocol- {}->{}".format(args_train.train_datasets, args_test.test_datasets))

    model = get_model(args_train).cuda()
    train_loader_stage1, train_loader_stage2, val_loaders = build_data_loader(args_train, args_test)
    criterion = make_loss(sum(args_train.classes))

    optimizer_prompt = make_optimizer_prompt(model, args_train)
    optimizer_prompt_domain = make_optimizer_prompt_domain(model, args_train)
    optimizer_image_encoder = make_optimizer_for_IE(model, args_train)

    scheduler_prompt = create_scheduler(args_train.prompt_epoch,args_train.prompt_lr, optimizer_prompt)
    scheduler_prompt_domain = create_scheduler(args_train.prompt_domain_epoch,args_train.prompt_lr, optimizer_prompt_domain)
    scheduler_image_encoder = WarmupMultiStepLR(optimizer_image_encoder, [30, 50], 0.1, 0.1, 10, 'linear')

    ### stage-1  ###
    train_stage_prior(train_loader_stage2=train_loader_stage2,
                      model=model,
                      criterion=criterion,
                      optimizer=optimizer_image_encoder,
                      scheduler=scheduler_image_encoder,
                      args_train=args_train,
                      logger_train=logger_train,
                      log_path=log_path,
                      epochs=args_train.prior_epoch)

    ### stage-2  ###
    train_stage1(train_loader_stage1=train_loader_stage1,
                 model=model,
                 optimizer=optimizer_prompt,
                 scheduler=scheduler_prompt,
                 args_train=args_train,
                 logger_train=logger_train,
                 log_path=log_path,
                 get_domain=False,
                 epochs=args_train.prompt_epoch)

    train_stage1(train_loader_stage1=train_loader_stage1,
                 model=model,
                 optimizer=optimizer_prompt_domain,
                 scheduler=scheduler_prompt_domain,
                 args_train=args_train,
                 logger_train=logger_train,
                 log_path=log_path,
                 get_domain=True,
                 epochs=args_train.prompt_domain_epoch)

    ### stage-3  ###
    train_stage2(train_loader_stage2=train_loader_stage2,
                 model=model,
                 criterion=criterion,
                 optimizer=optimizer_image_encoder,
                 scheduler=scheduler_image_encoder,
                 testloaders=val_loaders,
                 args_train=args_train,
                 logger_train=logger_train,
                 logger_test=logger_test,
                 log_path=log_path,
                 epochs=args_train.image_encoder_epoch)

    test(val_loaders, model, logger_test)
