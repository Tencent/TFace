import os.path as path
import logging
import torch
from ..head.distfc.partial_fc import PartialFC


class CkptLoader(object):
    @staticmethod
    def load_backbone(backbone, backbone_resume, local_rank):
        """ load pretrain backbone checkpoint
        """
        if not path.isfile(backbone_resume):
            logging.info("Backbone checkpoint %s not exists" % backbone_resume)
        else:
            # For DDP trained model, it should specify map_location device in load
            device = torch.device("cuda:%d" % local_rank)
            backbone.load_state_dict(torch.load(backbone_resume, map_location=device))
            logging.info("Loading backbone checkpoint %s succeed" % backbone_resume)

    @staticmethod
    def load_head(heads, head_resume, dist_fc, rank):
        """ load pretrain head checkpoint
        """
        if dist_fc:
            head_resume = '%s_Split_%d_checkpoint.pth' % (head_resume, rank)
        if not path.isfile(head_resume):
            logging.info('Head checkpoint %s not exists' % head_resume)
        else:
            pretrain_heads = torch.load(head_resume)
            for name, head in heads.items():
                # If pretrain heads do not contain branch, skip it
                if name not in pretrain_heads:
                    continue
                if isinstance(head, PartialFC):
                    head.load_pretrain_weight(pretrain_heads[name])
                else:
                    head.load_state_dict(pretrain_heads[name])
            logging.info("Loading head checkpoint %s succeed" % head_resume)

    @staticmethod
    def load_meta(opt, scaler, task, meta_resume):
        if not path.isfile(meta_resume):
            logging.info('Meta checkpoint %s not exists' % meta_resume)
        else:
            meta_dict = torch.load(meta_resume)
            if scaler and meta_dict.get('AMP_SCALER', None):
                scaler.load_state_dict(meta_dict['AMP_SCALER'])
            task.start_epoch = meta_dict['EPOCH']

            if isinstance(opt, dict):
                backbone_opt = opt['backbone']
                backbone_opt.load_state_dict(meta_dict['BACKBONE_OPT'])
            else:
                opt.load_state_dict(meta_dict['OPTIMIZER'])

            logging.info("Loading meta Checkpoint %s succeed" % meta_resume)


class CkptSaver(object):
    @staticmethod
    def save_backbone(backbone, model_root, epoch, rank):
        """ save backbone ckpt
        """
        if rank == 0:
            backbone_path = path.join(model_root, "Backbone_Epoch_%d_checkpoint.pth" % epoch)
            torch.save(backbone.module.state_dict(), backbone_path)

    @staticmethod
    def save_heads(heads, model_root, epoch, dist_fc, rank):
        """ save heads, if dist_fc is True, the head should be splited
        """
        head_dict = {}
        for name, head in heads.items():
            if isinstance(head, PartialFC):
                head_dict[name] = head.weight.data
            else:
                head_dict[name] = head.state_dict()
        if dist_fc:
            head_path = path.join(model_root, "HEAD_Epoch_%d_Split_%d_checkpoint.pth" % (epoch, rank))
            torch.save(head_dict, head_path)
        elif rank == 0:
            head_path = path.join(model_root, "HEAD_Epoch_%d_checkpoint.pth" % epoch)
            torch.save(head_dict, head_path)

    @staticmethod
    def save_meta(meta_dict, model_root, epoch, rank):
        """ save optimizer and scaler
        """
        if rank == 0:
            opt_path = path.join(model_root, "META_Epoch_%d_checkpoint.pth" % epoch)
            torch.save(meta_dict, opt_path)
