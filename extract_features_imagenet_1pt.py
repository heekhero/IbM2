import argparse
import pickle

import torch.backends.cudnn
import torch.optim
import torch.utils.data

from config import PATH, IMAGENET_1PT_PATH
from datasets.metaset import MetaSet
from model.dino_vision_transformer import vit_small
from model.mocov3_vits import vit_small as mocov3_vit_small
from model.msn_deit import deit_small, deit_large_p7, deit_base_p4
from model.resnet import resnet50
from utils import *

torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_test', default=True, action='store_true')
    parser.add_argument('--arch', default='deit_small_p16', choices=['deit_small_p16', 'deit_large_p7', 'deit_base_p4', 'resnet50'])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--pretrain_method', default='MoCov3', choices=['DINO', 'MSN', 'MoCov3', 'SimCLR', 'BYOL'])
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--checkpoint', default='checkpoint')

    args = parser.parse_args()

    args.dataset = 'Imagenet_1pt'
    args.exp_dir = os.path.join(PATH, args.checkpoint, args.pretrain_method, args.dataset, args.arch, 'features')

    if args.pretrain_method == 'MSN':
        if args.arch == 'deit_small_p16':
            args.load_path = 'checkpoint/MSN/vits16_800ep.pth.tar'
        elif args.arch == 'deit_large_p7':
            args.load_path = 'checkpoint/MSN/vitl7_200ep.pth.tar'
        elif args.arch == 'deit_base_p4':
            args.load_path = 'checkpoint/MSN/vitb4_300ep.pth.tar'
        else:
            raise NotImplementedError
    elif args.pretrain_method == 'DINO':
        if args.arch == 'deit_small_p16':
            args.load_path = 'checkpoint/DINO/dino_deitsmall16_pretrain.pth'
        else:
            raise NotImplementedError
    elif args.pretrain_method == 'MoCov3':
        if args.arch == 'deit_small_p16':
            args.load_path = 'checkpoint/MoCov3/vit-s-300ep.pth.tar'
        else:
            raise NotImplementedError
    elif args.pretrain_method == 'SimCLR':
        if args.arch == 'resnet50':
            args.load_path = 'checkpoint/SimCLR/model_final_checkpoint_phase999.torch'
        else:
            raise NotImplementedError
    elif args.pretrain_method == 'BYOL':
        if args.arch == 'resnet50':
            args.load_path = 'checkpoint/BYOL/byol_resnet50_8xb32-accum16-coslr-300e_in1k_20220225-a0daa54a.pth'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    args.load_path = os.path.join(PATH, args.load_path)
    print(args)

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    mset = MetaSet(data_path=IMAGENET_1PT_PATH)

    if args.pretrain_method == 'MSN':
        if args.arch == 'deit_small_p16':
            model = deit_small()
        elif args.arch == 'deit_large_p7':
            model = deit_large_p7()
        elif args.arch == 'deit_base_p4':
            model = deit_base_p4()
        else:
            raise NotImplementedError
    elif args.pretrain_method == 'DINO':
        if args.arch == 'deit_small_p16':
            model = vit_small(patch_size=16)
        elif args.arch == 'deit_small_p8':
            model = vit_small(patch_size=8)
        else:
            raise NotImplementedError
    elif args.pretrain_method == 'MoCov3':
        if args.arch == 'deit_small_p16':
            model = mocov3_vit_small(num_classes=0)
        else:
            raise NotImplementedError
    elif args.pretrain_method == 'iBOT':
        if args.arch == 'deit_small_p16':
            model = iBOT_vit_small(patch_size=16)
        else:
            raise NotImplementedError
    elif args.pretrain_method in ['BYOL', 'SimCLR', 'SwAV']:
        assert args.arch == 'resnet50'
        model = resnet50()
    else:
        raise NotImplementedError

    print(model)

    load_dict = torch.load(args.load_path, map_location=torch.device('cpu'))
    if args.pretrain_method == 'MSN':
        load_dict = {k.replace('module.', ''): v for k, v in load_dict['target_encoder'].items() if k.replace('module.', '') in model.state_dict()}
    elif args.pretrain_method == 'DINO':
        load_dict = {k.replace('module.', ''): v for k, v in load_dict.items() if k.replace('module.', '') in model.state_dict() and ('head' not in k)}
    elif args.pretrain_method == 'MoCov3':
        load_dict = {k.replace('module.', ''): v for k, v in load_dict['model'].items() if k.replace('module.', '') in model.state_dict()}
        if args.arch == 'deit_small_p16':
            assert len(load_dict) == 150
        else:
            raise NotImplementedError
    elif args.pretrain_method == 'iBOT':
        load_dict = {k.replace('module.', ''): v for k, v in load_dict['state_dict'].items() if k.replace('module.', '') in model.state_dict()}
        if args.arch == 'deit_small_p16':
            assert len(load_dict) == 150
        else:
            raise NotImplementedError

    elif args.pretrain_method == 'SimCLR':
        st = torch.load(args.load_path, map_location=torch.device('cpu'))['classy_state_dict']['base_model']['model']['trunk']
        load_dict = {k.replace('_feature_blocks.', ''): v for k, v in st.items()}
        print('len of load_dict is {}'.format(len(load_dict)))

    elif args.pretrain_method == 'BYOL':
        st = torch.load(args.load_path, map_location=torch.device('cpu'))['state_dict']
        load_dict = {k.replace('module.', ''): v for k, v in st.items()}
        assert len(load_dict) == 318

    elif args.pretrain_method == 'SwAV':
        state_dict = torch.load(args.load_path, map_location=torch.device('cpu'))

        new_state_dict = {}
        for k, v in state_dict.items():
            if ('projection' not in k) and ('prototypes' not in k):
                new_state_dict[k.replace('module.', '')] = v

        load_dict = new_state_dict
        print('len of load_dict is {}'.format(len(load_dict)))

    else:
        raise NotImplementedError

    if args.arch == 'deit_small_p16':
        assert len(load_dict) == 150

    print('len of load dict {}'.format(len(load_dict)))

    model.load_state_dict(load_dict, strict=True)

    model.cuda()
    model.eval()

    train_plain = mset.train_plain
    dloader = torch.utils.data.DataLoader(
        train_plain,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers)
    save_path_all = os.path.join(args.exp_dir, 'train_1pt.pth')

    feature_list = []
    target_list = []
    with torch.no_grad():
        for images, targets, _ in tqdm(dloader):

            images = images.float().cuda(non_blocking=True)

            features = F.normalize(model(images).cpu(), dim=-1)
            assert abs(features.pow(2).sum(dim=-1).sum() - features.size(0)) < 1e-4

            feature_list.append(features.detach().cpu())
            target_list.append(targets.detach().cpu())


        features_gather = torch.cat(feature_list, dim=0)
        targets_gather = torch.cat(target_list, dim=0)
        print('average of magnitude of train features is {:.4f}'.format(torch.abs(features_gather).mean().item()))

    feature_dict = {}
    feature_dict['features'] = features_gather
    feature_dict['targets'] = targets_gather
    print('len of features', len(features_gather))
    with open(save_path_all, 'wb') as f:
        pickle.dump(feature_dict, f)

    if args.save_test:
        dloader_test = torch.utils.data.DataLoader(
            mset.test,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers)
        save_path_test = os.path.join(args.exp_dir, 'test.pth')

        feature_list = []
        target_list = []
        with torch.no_grad():
            for images, targets, _ in tqdm(dloader_test):

                images = images.float().cuda(non_blocking=True)

                features = F.normalize(model(images).cpu(), dim=-1)
                assert abs(features.pow(2).sum(dim=-1).sum() - features.size(0)) < 1e-4

                feature_list.append(features.detach().cpu())
                target_list.append(targets.detach().cpu())


            features_gather = torch.cat(feature_list, dim=0)
            targets_gather = torch.cat(target_list, dim=0)
            print('average of magnitude of test features is {:.4f}'.format(torch.abs(features_gather).mean().item()))

        feature_dict = {}
        feature_dict['features'] = features_gather
        feature_dict['targets'] = targets_gather
        print('len of features', len(features_gather))
        with open(save_path_test, 'wb') as f:
            pickle.dump(feature_dict, f)



if __name__ == '__main__':
    main()