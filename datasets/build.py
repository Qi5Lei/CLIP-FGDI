import torch
from . import DATASET_REGISTRY
from .common import CommDataset
from torch.utils.data import DataLoader
from .samplers.sampler import RandomIdentitySampler
from .trans import bulid_transforms


def collate_fn(batch):
    imgs, pids, camids, viewids, image_path, domains, cid = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    cid = torch.tensor(cid, dtype=torch.int64)
    domains = torch.tensor(domains, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, domains, cid




def build_data_loader(args, args_test):
    train_items = list()
    for d in args.train_datasets:
        dataset = DATASET_REGISTRY.get(d)(root=args.data_path, combineall=args.combine_all)
        train_items.extend(dataset.train)

    train_transforms = bulid_transforms(args, is_train=True)

    train_set = CommDataset(train_items, train_transforms)
    num_workers = args.num_workers

    sampling_method = RandomIdentitySampler(train_items,args.batch_size, 4)

    train_loader_stage_2 = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      sampler=sampling_method,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn)


    val_transforms = bulid_transforms(args_test,is_train=False)

    train_set_normal = CommDataset(train_items, val_transforms)
    train_loader_stage_1 = DataLoader(
        train_set_normal, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn
    )

    dataset_names = args_test.test_datasets
    val_loaders = {}
    for elm in dataset_names:
        dataset = DATASET_REGISTRY.get(elm)(root=args.data_path)
        test_items = dataset.query + dataset.gallery

        val_set = CommDataset(test_items, val_transforms, relabel=False)

        val_loader = DataLoader(
            val_set, batch_size=args_test.test_batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn
        )
        val_loaders[elm] = [val_loader, len(dataset.query)]

    return train_loader_stage_1, train_loader_stage_2, val_loaders




def build_data_loaders(args, args_test, model):
    val_transforms = bulid_transforms(args,is_train=False)
    train_items = list()
    trainstage1_items = []
    trainstage1_sets = {}
    max_id = 0
    for i,d in enumerate(args.train_datasets):
        dataset = DATASET_REGISTRY.get(d)(root=args.data_path, combineall=args.combine_all)
        train_items.extend(dataset.train)
        trainstage1_items.append(dataset.train)
        trainstage1_sets[d] = CommDataset(dataset.train,val_transforms,last_id=max_id)
        max_id+=args.classes[i]

    train_transforms = bulid_transforms(args,is_train=True)

    train_set = CommDataset(train_items, train_transforms)
    num_workers = args.num_workers

    sampling_method = RandomIdentitySampler(train_items,args.batch_size, 4)

    train_loader_stage_2 = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      sampler=sampling_method,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn)


    train_loader_stage_1 = {d:DataLoader(
        trainstage1_set, batch_size=args.batch_size//4, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn
    ) for d,trainstage1_set in trainstage1_sets.items()}

    dataset_names = args_test.test_datasets
    val_loaders = {}
    for elm in dataset_names:
        dataset = DATASET_REGISTRY.get(elm)(root=args.data_path)
        test_items = dataset.query + dataset.gallery

        val_set = CommDataset(test_items, val_transforms, relabel=False)

        val_loader = DataLoader(
            val_set, batch_size=args_test.test_batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn
        )
        val_loaders[elm] = [val_loader, len(dataset.query)]
    return train_loader_stage_1, train_loader_stage_2, val_loaders
