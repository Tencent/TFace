import torch
import torch.nn as nn
import time
import os
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from models import *
from datasets import *
from utils import *


class Solver(object):
    def __init__(self, args):
        """Init the global settings including device, seed, models, dataloaders, crterions, optimizers and schedulers

        Args:
            args
        """
        super(Solver, self).__init__()

        self.args = args
        self.log_path = os.path.join(self.args.output_path, "log.txt")
        self.emotions = ["hap", "sad", "neu", "ang", "sur", "dis", "fea"]
        self.best_wa = 0
        self.best_ua = 0

        # init cuda
        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])
        self.device = torch.device(
            'cuda:%d' % self.args.gpu_ids[0] if self.args.gpu_ids else 'cpu')

        # set seed
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # init model
        self.model = create_model(self.args)
        if len(self.args.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, self.args.gpu_ids)
        self.model.to(self.device)

        # init dataloader
        self.train_dataloader = create_dataloader(self.args, "train")
        self.test_dataloader = create_dataloader(self.args, "test")

        # init criterion
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing).to(self.device)

        # init optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.args.lr,
                                           eps=self.args.eps,
                                           weight_decay=self.args.weight_decay)
        self.scheduler = build_scheduler(
            self.args, self.optimizer, len(self.train_dataloader))

        # resume
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cuda:0',  weights_only=True)
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            self.args.start_epoch = checkpoint['epoch'] + 1
            self.best_wa = checkpoint['best_wa']
            self.best_ua = checkpoint['best_ua']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def run(self):

        for epoch in range(self.args.start_epoch, self.args.epochs):
            inf = '********************' + str(epoch) + '********************'
            start_time = time.time()

            with open(self.log_path, 'a') as f:
                f.write(inf + '\n')
            print(inf)

            # train the model for one epoch
            train_acc, train_loss = self.train(epoch)
            # validate the model
            val_acc, val_loss = self.validate(epoch)

            # remember best acc and save checkpoint
            is_best = (val_acc[0] > self.best_wa) or (
                val_acc[1] > self.best_ua)
            self.best_wa = max(val_acc[0], self.best_wa)
            self.best_ua = max(val_acc[1], self.best_ua)
            self.save({'epoch': epoch,
                       'state_dict': self.model.state_dict(),
                       'best_wa': self.best_wa,
                       'best_ua': self.best_ua,
                       'optimizer': self.optimizer.state_dict(),
                       'args': self.args}, is_best)

            # print and save log
            epoch_time = time.time() - start_time
            msg = self.get_acc_msg(epoch, train_acc, train_loss, val_acc, val_loss,
                                   self.best_wa, self.best_ua, epoch_time)
            with open(self.log_path, 'a') as f:
                f.write(msg)
            print(msg)

            if is_best:
                # print confusion matrix
                cm_msg = self.get_confusion_msg(val_acc[2])
                with open(self.log_path, 'a') as f:
                    f.write(cm_msg)
                print(cm_msg)

                # convert confusion matrix to heatmap
                cm = []
                for row in val_acc[2]:
                    row = row / np.sum(row)
                    cm.append(row)
                fig_path = os.path.join(self.args.output_path, "fig_best.png")
                ax = seaborn.heatmap(
                    cm, xticklabels=self.emotions, yticklabels=self.emotions, cmap='rocket_r')
                figure = ax.get_figure()
                # save the heatmap
                figure.savefig(fig_path)
                plt.close()

        return self.best_ua, self.best_ua

    def train(self, epoch):
        """ Train the model for one eopch
        """
        self.model.train()

        all_pred, all_target = [], []
        all_loss = 0

        for i, (images, target) in enumerate(self.train_dataloader):

            print("Training epoch \t{}: {}\\{}".format(
                epoch, i + 1, len(self.train_dataloader)), end='\r')

            images = images.to(self.device)
            target = target.to(self.device)

            output = self.model(images)

            loss = self.criterion(output, target)

            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target = target.cpu().numpy()

            all_pred.extend(pred)
            all_target.extend(target)
            all_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step_update(epoch * len(self.train_dataloader) + i)

        # WAR
        acc1 = accuracy_score(all_target, all_pred)

        # UAR
        acc2 = balanced_accuracy_score(all_target, all_pred)

        loss = all_loss / len(self.train_dataloader)

        return [acc1, acc2], loss

    def validate(self, epoch):
        """Validate the model for one epoch
        """
        self.model.eval()

        all_pred, all_target = [], []
        all_loss = 0

        for i, (images, target) in enumerate(self.test_dataloader):

            print("Testing epoch \t{}: {}\\{}".format(
                epoch, i + 1, len(self.test_dataloader)), end='\r')

            images = images.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = self.model(images)

            loss = self.criterion(output, target)

            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target = target.cpu().numpy()

            all_pred.extend(pred)
            all_target.extend(target)
            all_loss += loss.item()

        # WAR
        acc1 = accuracy_score(all_target, all_pred)

        # UAR
        acc2 = balanced_accuracy_score(all_target, all_pred)

        c_m = confusion_matrix(all_target, all_pred)
        loss = all_loss / len(self.test_dataloader)

        return [acc1, acc2, c_m], loss

    def save(self, state, is_best):
        # save the best model
        if is_best:
            checkpoint_path = os.path.join(
                self.args.output_path, "model_best.pth")
            torch.save(state, checkpoint_path)

        # save the latest model for resume
        checkpoint_path = os.path.join(
            self.args.output_path, "model_latest.pth")
        torch.save(state, checkpoint_path)

    def get_acc_msg(self, epoch, train_acc, train_loss, val_acc, val_loss, best_wa, best_ua, epoch_time):
        msg = """\nEpoch {} Train\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                   Epoch {} Test\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                   Epoch {} Best\t: WA:{:.2%}, \tUA:{:.2%}
                   Epoch {} Time\t: {:.1f}s\n\n""".format(epoch, train_acc[0], train_acc[1], train_loss, 
                                                          epoch, val_acc[0], val_acc[1], val_loss, 
                                                          epoch, best_wa, best_ua, epoch, epoch_time)
        return msg

    def get_confusion_msg(self, confusion_matrix):
        # change the format of cunfusion matrix to print
        msg = "Confusion Matrix:\n"
        for i in range(len(confusion_matrix)):
            msg += self.emotions[i]
            for cell in confusion_matrix[i]:
                msg += "\t" + str(cell)
            msg += "\n"
        for emotion in self.emotions:
            msg += "\t" + emotion
        msg += "\n\n"
        return msg
