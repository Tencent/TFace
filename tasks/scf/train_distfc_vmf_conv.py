import logging
import torch
import torch.cuda.amp as amp
import torch.distributed as dist

from torchkit.util.utils import AverageMeter, Timer
from torchkit.util.utils import adjust_learning_rate, warm_up_lr
from torchkit.util.utils import accuracy_dist
from torchkit.util.distributed_functions import AllGather
from torchkit.loss import get_loss
from torchkit.task.base_task import BaseTask

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


class TrainTask(BaseTask):
    """TrainTtask in model distributed mode, which means classifier shards into multi workers
    """
    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)


    
    def _loop_step_vmf(self, train_loaders, backbone, uncer_net,  kl, fc_ckpt, opt,
                   scaler, epoch, class_splits):
        log_step = 100  # 100 batch
        # backbone.train()  # set to training mode
        # for head in heads:
            # head.train()
        uncer_net.train()
        batch_sizes = self.batch_sizes

        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for batch, samples in enumerate(zip(*train_loaders)):
            global_batch = epoch * self.step_per_epoch + batch
            if global_batch <= self.warmup_step:
                warm_up_lr(global_batch, self.warmup_step, self.cfg['LR'], opt)
            if batch >= self.step_per_epoch:
                break

            inputs = torch.cat([x[0] for x in samples], dim=0)
            inputs = inputs.cuda(non_blocking=True)
            labels = torch.cat([x[1] for x in samples], dim=0)
            labels = labels.cuda(non_blocking=True)

            with torch.no_grad():
                features, conv_features = backbone(inputs)

            mu_norm = torch.norm(features, dim=1, keepdim=True) # --> [2B,]
            mu = features / mu_norm # mu \in S^{d-1}

            log_kappa = uncer_net(conv_features)
            kappa = torch.exp(log_kappa)
            kappa_mean = kappa.mean()

            labels = torch.split(labels, split_size_or_sections=batch_sizes) #for x in labels
            kappa = torch.split(kappa, split_size_or_sections=batch_sizes) #for x in kappa
            mu = torch.split(mu, split_size_or_sections=batch_sizes) #for x in mu 

            mus = []
            for x in mu:
                mus.append(x)

            kappas = []
            for x in kappa:
                kappas.append(x)
            all_labels = []
            wcs = []
            #for i in range(len(batch_sizes)):
            #    all_labels.append(torch.cat([x[i] for x in labels], dim=0).cuda())
            for x in labels:
                all_labels.append(x)

            step_losses = []
            # step_original_outputs = []
            for i in range(len(batch_sizes)):
                label = all_labels[i]
                wc = fc_ckpt[label, :]
                loss, l1, l2, l3 = kl(mus[i], kappas[i], wc)
                loss = loss.mean()
                step_losses.append(loss)

            total_loss = sum(step_losses)
            opt.zero_grad()
            total_loss.backward()
            opt.step()


            for i in range(len(batch_sizes)):
                # measure accuracy and record loss
                am_losses[i].update(step_losses[i].data.item(),
                                    all_labels[i].size(0))
               
                if self.rank == 0 and (batch == 0 or ((batch + 1) % log_step == 0)):
                    summarys = {
                        'train/loss_%d' % i: am_losses[i].val
                        
                    }
                    self._writer_summarys(summarys, batch, epoch)

            duration = t.get_duration()
            # dispaly training loss & acc every DISP_FREQ
            if self.rank == 0 and (batch == 0 or ((batch + 1) % log_step == 0)):
                self._log_tensor_loss(batch, epoch, duration, am_losses)

    def train(self):
        train_loaders, class_nums = self._make_inputs()
        backbone, heads, class_splits = self._make_model(class_nums)
        ucertnet = self._make_uncer_net_conv()


        self._load_pretrain_model(backbone, self.cfg['BACKBONE_RESUME'], heads, self.cfg['HEAD_RESUME'])

        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, device_ids=[self.local_rank])

        ucertnet = torch.nn.parallel.DistributedDataParallel(ucertnet, device_ids=[self.local_rank])
        # loss = get_loss(self.cfg['LOSS_NAME']).cuda()
        kl_loss = self._get_kl_loss(512, 64)
        # opt = self._get_optimizer(backbone, heads)
        opt = self._get_optimizer3(ucertnet)

        fc = self._load_fc_model(self.cfg['FC_RESUME_ROOT'])
        scaler = amp.GradScaler()
        self._load_meta(opt, scaler, self.cfg['META_RESUME'])
        self._create_writer()
        for epoch in range(self.start_epoch, self.epoch_num):
            adjust_learning_rate(opt, epoch, self.cfg)
            # self._loop_step(train_loaders, backbone, heads, loss, opt, scaler, epoch, class_splits)
            self._loop_step_vmf(train_loaders, backbone, ucertnet,  kl_loss, fc, opt,
                   scaler, epoch, class_splits)
            self._save_ckpt(epoch, backbone, heads, opt, scaler)
            self._save_ckpt2(epoch, ucertnet, 'uncertain')


def main():
    task = TrainTask('train_config_dist.yaml')
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
