import os
import sys
import torch
import torch.nn.init as init
from tasks.duetface.duetface_model import DuetFaceModel

from ..head import get_head
from ..util import get_class_split

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

# ================================================
# do NOT run this script directly
# append the following make_interactive_models() function to base_task.py
# right after make_models()
# ================================================


def make_interactive_models(self):
    sub_channels = self.cfg['SUB_CHS']
    main_feature_size = self.cfg['FEATURE_SIZE']
    sub_feature_size = self.cfg['SUB_FEATURE_SIZE']

    num_sub_channels = len(sub_channels) * 3
    self.backbone = DuetFaceModel(num_sub_channels, main_feature_size, sub_feature_size,
                                  main_model_name=self.cfg['BACKBONE_NAME'],
                                  sub_model_name=self.cfg['SUB_BACKBONE_NAME'])

    # load pre-trained client-side model from a given checkpoint
    if self.cfg['LOAD_CKPT']:
        ckpt_path = self.cfg['CKPT_PATH']
        if not os.path.exists(ckpt_path):
            raise RuntimeError("%s not exists" % ckpt_path)

        model_dict = self.backbone.sub_model.state_dict()
        pretrained_dict = torch.load(ckpt_path, weights_only=True)
        pretrained_dict = {k.split('.', 1)[1]: v for k, v in pretrained_dict.items() if
                           k.split('.', 1)[1] in model_dict}

        model_dict.update(pretrained_dict)
        self.backbone.sub_model.load_state_dict(model_dict)

    self.backbone = self.backbone.cuda()
    # logging.info("DuetFace Backbone Generated")

    # make heads, the rest are identical to that in make_model()
    embedding_size = self.cfg['EMBEDDING_SIZE']
    self.class_shards = []
    metric = get_head(self.cfg['HEAD_NAME'], dist_fc=self.dist_fc)

    for name, branch in self.branches.items():
        class_num = self.class_nums[name]
        class_shard = get_class_split(class_num, self.world_size)
        self.class_shards.append(class_shard)
        # logging.info('Split FC: {}'.format(class_shard))

        init_value = torch.FloatTensor(embedding_size, class_num)
        init.normal_(init_value, std=0.01)
        head = metric(in_features=embedding_size,
                      gpu_index=self.rank,
                      weight_init=init_value,
                      class_split=class_shard,
                      scale=branch.scale,
                      margin=branch.margin)
        del init_value
        head = head.cuda()
        self.heads[name] = head
