# -*- coding: utf-8 -*-
import cv2
import time
import pickle
import os
import shutil
from numpy import *
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel

from models.networks import *
from models.utils import *
from models.Pretrained_FR_Models import irse, facenet, ir152


# *************************
# Main module of Adv-Makeup
# *************************
class MakeupAttack(nn.Module):
    def __init__(self, config):
        super(MakeupAttack, self).__init__()

        self.config = config
        self.lr = config.lr
        self.lr_discr = self.lr / 2.5
        self.update_lr_m = config.update_lr_m

        self.device = torch.device('cuda:{}'.format(config.gpu)) if config.gpu >= 0 else torch.device('cpu')
        self.data_dir = config.data_dir
        self.api_landmarks = pickle.load(open(self.data_dir + '/' + config.lmk_name, 'rb'))
        self.log_dir = config.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.resize_size = config.resize_size
        self.eye_area = config.eye_area
        self.mean = config.mean
        self.std = config.std

        # Discriminator
        self.discr = init_net(Discriminator(config.input_dim), self.device, init_type='normal')

        # Encoder
        self.enc = Encoder(config.input_dim).to(self.device)

        # Decoder
        self.dec = Decoder(config.input_dim).to(self.device)

        # Optmizers for discriminator
        self.discr_opt = torch.optim.Adam(self.discr.parameters(),
                                          lr=self.lr_discr, betas=(0.5, 0.999), weight_decay=0.0001)
        # Optmizers for generator
        self.gen_opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()) ,
                                        lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # Criterion
        self.criterionL1 = nn.L1Loss()

        # VGG16 pretrained model
        self.vgg16 = Vgg16().to(self.device)
        self.mean_shift = MeanShift(self.device)

        # FR model
        self.models_info = {}
        # White-box FR models' name list
        self.train_model_name_list = config.train_model_name_list
        # Black-box FR models' name list
        self.val_model_name_list = config.val_model_name_list
        for model_name in self.train_model_name_list + self.val_model_name_list:
            self.models_info[model_name] = [[], []]
            if model_name == 'ir152':
                self.models_info[model_name][0].append((112, 112))
                self.fr_model = ir152.IR_152((112, 112))
                self.fr_model.load_state_dict(torch.load('./models/Pretrained_FR_Models/ir152.pth', weights_only=True))
            if model_name == 'irse50':
                self.models_info[model_name][0].append((112, 112))
                self.fr_model = irse.Backbone(50, 0.6, 'ir_se')
                self.fr_model.load_state_dict(torch.load('./models/Pretrained_FR_Models/irse50.pth', weights_only=True))
            if model_name == 'mobile_face':
                self.models_info[model_name][0].append((112, 112))
                self.fr_model = irse.MobileFaceNet(512)
                self.fr_model.load_state_dict(torch.load('./models/Pretrained_FR_Models/mobile_face.pth', weights_only=True))
            if model_name == 'facenet':
                self.models_info[model_name][0].append((160, 160))
                self.fr_model = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
                self.fr_model.load_state_dict(torch.load('./models/Pretrained_FR_Models/facenet.pth', weights_only=True))
            self.fr_model.to(self.device)
            self.fr_model.eval()
            self.models_info[model_name][0].append(self.fr_model)

    '''
    Model Initialization
    '''
    def res_init(self, ep):
        # Cumulative Losses
        self.disp_ad_true_loss = 0
        self.disp_ad_fake_loss = 0
        self.disp_g_loss = 0
        self.tar_l = 0
        self.style_l = 0
        self.grad_l = 0
        self.tv_l = 0

        # Res Image
        self.gen_img =None
        self.orig_before = None
        self.orig_after = None

        # Reset Optim
        if (ep + 1) % 100 == 0:
            self.lr_discr /= 10
            self.lr /= 10
            self.discr_opt = torch.optim.Adam(self.discr.parameters(), lr=self.lr_discr, betas=(0.5, 0.999),
                                              weight_decay=0.0001)
            self.gen_opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.lr,
                                            betas=(0.5, 0.999), weight_decay=0.0001)

    '''
    Targeted Loss Calculation
    '''
    def cos_simi(self, emb_before_pasted, emb_target_img):
        """
        :param emb_before_pasted: feature embedding for the generated adv-makeup face images
        :param emb_target_img: feature embedding for the victim target image
        :return: cosine similarity between two face embeddings
        """
        return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                          / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))

    def cal_target_loss(self, before_pasted, target_img, model_name):
        """
        :param before_pasted: generated adv-makeup face images
        :param target_img: victim target image
        :param model_name: FR model for embedding calculation
        :return: cosine distance between two face images
        """

        # Obtain model input size
        input_size = self.models_info[model_name][0][0]
        # Obtain FR model
        fr_model = self.models_info[model_name][0][1]

        before_pasted_resize = F.interpolate(before_pasted, size=input_size, mode='bilinear')
        target_img_resize = F.interpolate(target_img, size=input_size, mode='bilinear')

        # Inference to get face embeddings
        emb_before_pasted = fr_model(before_pasted_resize)
        emb_target_img = fr_model(target_img_resize).detach()

        # Cosine loss computing
        cos_loss = 1 - self.cos_simi(emb_before_pasted, emb_target_img)

        return cos_loss

    '''
    Paste makeup to the eyearea
    '''
    def paste_patch(self, before, before_path, target_name):
        """
        :param before: generated eye-area images with adv. makeup
        :param before_path: Un-makeup images' pathes
        :param target_name: file name of the victim target image
        :return: a set of pasting results
        """
        before_orig = []
        before_pasted = []
        target_img = []
        fake_afters = []
        before_pasted_eyes  = []
        for i, before_name in enumerate(before_path):
            img = read_img_from_path(self.data_dir, before_name, self.mean, self.mean, self.device)

            lmks = self.api_landmarks[before_name].astype(int)

            # Obtain the coordinates of the top-left and bottom right corners of the eye_area
            top_left = [min(lmks[self.eye_area, 0]), min(lmks[self.eye_area, 1])]
            bottom_right = [max(lmks[self.eye_area, 0]), max(lmks[self.eye_area, 1])]

            # Obtain the width and height of the eye-area
            orig_h = bottom_right[1] - top_left[1]
            orig_w = bottom_right[0] - top_left[0]

            # Obtain the coordinates surrounding the polygons of two eyes
            left_eye_polygon = [(l[0]-top_left[0], l[1]-top_left[1]) for
                                l in lmks[[94, 1, 34, 53, 59, 67, 3, 12], :]]
            right_eye_polygon = [(l[0]-top_left[0], l[1]-top_left[1]) for
                                 l in lmks[[27, 104, 41, 85, 20, 47, 43, 51], :]]

            # Obtain the coordinates lager than the polygons of two eyes
            left_region_polygon = [(l[0]-top_left[0], l[1]-top_left[1] if j < 6 else l[1]-top_left[1] + 40)
                                   for j, l in enumerate(lmks[[19, 84, 29, 79, 28, 35, 59, 67, 3, 12, 94], :])]
            right_region_polygon = [(l[0]-top_left[0], l[1]-top_left[1] if j < 6 else l[1]-top_left[1] + 40)
                                    for j, l in enumerate(lmks[[91, 24, 73, 70, 75, 74, 20, 47, 43, 51, 27], :])]

            img_i = Image.new('L', (orig_w, orig_h), 0)
            ImageDraw.Draw(img_i).polygon(left_region_polygon, outline=1, fill=1)
            ImageDraw.Draw(img_i).polygon(right_region_polygon, outline=1, fill=1)
            ImageDraw.Draw(img_i).polygon(left_eye_polygon, outline=0, fill=0)
            ImageDraw.Draw(img_i).polygon(right_eye_polygon, outline=0, fill=0)
            mask = asarray(img_i)
            # Obtain the 0-1 mask for the orbital eye regions
            mask = torch.from_numpy(mask).to(torch.float32).to(self.device)

            # Paste the generated eye makeup to the un-makeup face images by using the above mask
            before_orig.append(img[:,:,top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].clone())
            img[:,:,top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
                F.interpolate(before[i].unsqueeze(0), size=(orig_h, orig_w), mode='bilinear') * mask + \
                img[:, :, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] * (1-mask)

            # Images with the pasted adversarial eye makeup
            before_pasted.append(img)
            # Victim target images
            target_img.append(read_img_from_path(self.data_dir, 'target_aligned_600/' + target_name,
                                                 self.mean, self.std, self.device))
            fake_afters.append(F.interpolate(before[i].unsqueeze(0), size=(orig_h, orig_w), mode='bilinear'))
            # Only eye-area images with pasted adv. makeup
            before_pasted_eyes.append(img[:,:,top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].clone())

        before_pasted = torch.cat(before_pasted)
        target_img = torch.cat(target_img)

        return before_orig, before_pasted, target_img, fake_afters, before_pasted_eyes

    '''
    Param updates
    '''
    def update_discr(self, before, after):
        """
        :param before: Un-makeup eye area images (attacker's)
        :param after: Real-world makeup eye area images
        :return: None
        """
        self.discr_opt.zero_grad()
        fake_content = self.enc(before)
        fake_after = self.dec(*fake_content)
        pred_fake = self.discr(fake_after.to(self.device))
        pred_real = self.discr(after.to(self.device))

        out_fake = torch.sigmoid(pred_fake)
        out_real = torch.sigmoid(pred_real)
        ad_true_loss = F.mse_loss(out_real, torch.ones_like(out_real, device=self.device))
        ad_fake_loss = F.mse_loss(out_fake, torch.zeros_like(out_fake, device=self.device))

        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.discr.parameters(), 5)
        self.discr_opt.step()

        self.disp_ad_true_loss += ad_true_loss.item()
        self.disp_ad_fake_loss += ad_fake_loss.item()

        self.orig_before = before
        self.orig_after = after
        self.gen_img = fake_after

    def update_gen(self, before, before_path, target_name):
        """
        :param before: Un-makeup eye area images (attacker's)
        :param before_path: Un-makeup images' pathes
        :param target_name: file name of the victim target image
        :return: None
        """
        model_name_list_meta_train = self.train_model_name_list[:2]
        model_name_list_meta_test = self.train_model_name_list[2:]
        self.gen_opt.zero_grad()

        fake_content = self.enc(before)
        fake_after = self.dec(*fake_content)
        fake_pred = self.discr(fake_after.to(self.device))
        fake_out = torch.sigmoid(fake_pred)
        before_orig, before_pasted, target_img, fake_afters, before_pasted_eyes = \
            self.paste_patch(fake_after, before_path, target_name)

        overall_tar_loss = []
        for model_name in model_name_list_meta_train:
            overall_tar_loss.append(self.cal_target_loss(before_pasted, target_img, model_name))

        targeted_loss_test = []
        # Meta-train
        for model_name in model_name_list_meta_train:
            tar_l_tr = self.cal_target_loss(before_pasted, target_img, model_name)
            grad_enc = torch.autograd.grad(tar_l_tr, self.enc.parameters(), retain_graph=True)
            grad_dec = torch.autograd.grad(tar_l_tr, self.dec.parameters(), retain_graph=True)
            fast_weights_enc = list(map(lambda p: p[1] - self.update_lr_m * p[0], zip(grad_enc, self.enc.parameters())))
            fast_weights_dec = list(map(lambda p: p[1] - self.update_lr_m * p[0], zip(grad_dec, self.dec.parameters())))

            # Meta-test
            fake_content_test = self.enc(before, fast_weights_enc)
            fake_after_test = self.dec(*fake_content_test, fast_weights_dec)
            _, before_pasted_test, _, _, _ = self.paste_patch(fake_after_test, before_path, target_name)
            tar_l_te = []
            for model_name in model_name_list_meta_test:
                tar_l_te.append(self.cal_target_loss(before_pasted_test, target_img, model_name))
            tar_l_te = torch.mean(torch.stack(tar_l_te))

            targeted_loss_test.append(tar_l_te)

        overall_tar_loss += targeted_loss_test
        overall_tar_loss = torch.mean(torch.stack(overall_tar_loss))


        ''' style loss '''
        style_loss = 0
        num = 0
        for b_o, b_p_e in zip(before_orig, before_pasted_eyes):
            target_features_style = self.vgg16(self.mean_shift(b_o))
            target_gram_style = [gram_matrix(y) for y in target_features_style]

            blend_features_style = self.vgg16(self.mean_shift(b_p_e))
            blend_gram_style = [gram_matrix(y) for y in blend_features_style]

            s_loss = 0
            for layer in range(len(blend_gram_style)):
                s_loss += F.mse_loss(blend_gram_style[layer], target_gram_style[layer])
            s_loss /= len(blend_gram_style)
            s_loss *= 10e4
            style_loss += s_loss
            num += 1
        style_loss /= num

        '''grad loss'''
        grad_loss = 0
        num = 0
        for b_o, b_p_e in zip(before_orig, before_pasted_eyes):
            gt_gradient = laplacian_filter_tensor(b_o, self.device)
            pred_gradient = laplacian_filter_tensor(b_p_e, self.device)
            for c in range(len(pred_gradient)):
                grad_loss += F.mse_loss(pred_gradient[c], gt_gradient[c])
                num += 1
        grad_loss /= num

        '''tv loss'''
        tv_loss = 0
        num = 0
        for b_p_e in before_pasted_eyes:
            tv_loss += (torch.mean(torch.abs(b_p_e[:, :, :, :-1] - b_p_e[:, :, :, 1:])) + \
                      torch.mean(torch.abs(b_p_e[:, :, :-1, :] - b_p_e[:, :, 1:, :])))
            num += 1
        tv_loss /= num

        '''discr loss'''
        discr_loss = F.mse_loss(fake_out, torch.ones_like(fake_out, device=self.device))

        loss_G = discr_loss + overall_tar_loss + style_loss + grad_loss + tv_loss
        loss_G.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.discr.parameters(), 5)
        self.gen_opt.step()

        self.disp_g_loss += loss_G.item()
        self.tar_l += overall_tar_loss.item()
        self.style_l += style_loss.item()
        self.grad_l += grad_loss.item()
        self.tv_l += tv_loss.item()

        self.orig_before = before
        self.gen_img = fake_after
        self.orig_before_pasted = before_pasted

    '''
    Results save and visualization
    '''
    def save_res_img(self, idx, before, before_path, simi_scores_dict, target_name):
        """
        :param idx: bacth id
        :param before: generated eye-area images with adv. makeup
        :param before_path: un-makeup images' pathes
        :param simi_scores_dict: dictionary for saving similarity scores
        :param target_name: file name of the victim target image
        :return: None
        """
        save_dir = self.log_dir + '/test_imgs' + '/' + target_name.split('.')[0]
        os.makedirs(save_dir, exist_ok=True)
        fake_content = self.enc(before, bn_training=False)
        fake_after = self.dec(*fake_content, bn_training=False)

        _, before_pasted, target_img, _, _ = self.paste_patch(fake_after, before_path, target_name)
        before_orig = []
        for before_name in before_path:
            img = read_img_from_path(self.data_dir, before_name, self.mean, self.std, self.device)
            before_orig.append(img)
        before_orig = torch.cat(before_orig)

        for model_name in self.train_model_name_list + self.val_model_name_list:
            emb_before_orig = self.models_info[model_name][0][1]\
                (F.interpolate(before_orig, size=self.models_info[model_name][0][0], mode='bilinear')).detach()
            emb_before_pasted = self.models_info[model_name][0][1]\
                (F.interpolate(before_pasted, size=self.models_info[model_name][0][0], mode='bilinear')).detach()
            emb_target_img = self.models_info[model_name][0][1]\
                (F.interpolate(target_img, size=self.models_info[model_name][0][0], mode='bilinear')).detach()
            cosine_similarity_1 = self.cos_simi(emb_before_orig, emb_target_img)
            cosine_similarity_2 = self.cos_simi(emb_before_pasted, emb_target_img)
            if model_name not in simi_scores_dict:
                simi_scores_dict[model_name] = []
            simi_scores_dict[model_name].append((cosine_similarity_2).item())

        torchvision.utils.save_image(before_pasted / 2 + 0.5,
            os.path.join(save_dir, before_path[0].split('/')[-1].split('.')[0] + '_adv.png'), nrow=1)

        print("%d images processed!" % (idx))

    def save_res_tmp_img(self, idx, before, before_path, target_name):
        """
        :param idx: bacth id
        :param before: generated eye-area images with adv. makeup
        :param before_path: un-makeup images' pathes
        :param target_name: file name of the victim target image
        :return: None
        """
        save_dir = self.log_dir + '/test_imgs_tmp' + '/' + target_name.split('.')[0]
        os.makedirs(save_dir, exist_ok=True)
        fake_content = self.enc(before, bn_training=False)
        fake_after = self.dec(*fake_content, bn_training=False)

        fake_after_orig = []
        before_orig = []
        for i, before_name in enumerate(before_path):
            lmks_t = self.api_landmarks[before_name].astype(int)

            top_left_t = [min(lmks_t[self.eye_area, 0]), min(lmks_t[self.eye_area, 1])]
            bottom_right_t = [max(lmks_t[self.eye_area, 0]), max(lmks_t[self.eye_area, 1])]

            orig_h_t = bottom_right_t[1] - top_left_t[1]
            orig_w_t = bottom_right_t[0] - top_left_t[0]

            left_eye_polygon_t = [(l[0] - top_left_t[0], l[1] - top_left_t[1]) for l in
                                lmks_t[[94, 1, 34, 53, 59, 67, 3, 12], :]]
            right_eye_polygon_t = [(l[0] - top_left_t[0], l[1] - top_left_t[1]) for l in
                                 lmks_t[[27, 104, 41, 85, 20, 47, 43, 51], :]]

            left_region_polygon_t = [(l[0] - top_left_t[0], l[1] - top_left_t[1] if j < 6 else l[1]
                - top_left_t[1] + 40) for j, l in enumerate(lmks_t[[19, 84, 29, 79, 28, 35, 59, 67, 3, 12, 94], :])]
            right_region_polygon_t = [(l[0] - top_left_t[0], l[1] - top_left_t[1] if j < 6 else l[1]
                - top_left_t[1] + 40) for j, l in enumerate(lmks_t[[91, 24, 73, 70, 75, 74, 20, 47, 43, 51, 27], :])]

            img_i_t = Image.new('L', (orig_w_t, orig_h_t), 0)
            ImageDraw.Draw(img_i_t).polygon(left_region_polygon_t, outline=1, fill=1)
            ImageDraw.Draw(img_i_t).polygon(right_region_polygon_t, outline=1, fill=1)
            ImageDraw.Draw(img_i_t).polygon(left_eye_polygon_t, outline=0, fill=0)
            ImageDraw.Draw(img_i_t).polygon(right_eye_polygon_t, outline=0, fill=0)
            mask = asarray(img_i_t)
            mask = torch.from_numpy(mask).to(torch.float32).to(self.device)

            fake_after_orig.append(F.interpolate(fake_after[i].unsqueeze(0),
                                                 size=(orig_h_t, orig_w_t), mode='bilinear'))
            before_orig.append(F.interpolate(before[i].unsqueeze(0),
                                             size=(orig_h_t, orig_w_t), mode='bilinear'))

        fake_after_orig = torch.cat(fake_after_orig)
        before_orig = torch.cat(before_orig)

        torchvision.utils.save_image(before_orig / 2 + 0.5,
            os.path.join(save_dir, before_path[0].split('/')[-1].split('.')[0] + '_before.png'), nrow=1)
        torchvision.utils.save_image(fake_after_orig / 2 + 0.5,
            os.path.join(save_dir, before_path[0].split('/')[-1].split('.')[0] + '_fake_after.png'), nrow=1)
        torchvision.utils.save_image(mask,
            os.path.join(save_dir, before_path[0].split('/')[-1].split('.')[0] + '_mask.png'), nrow=1)

        print("%d images processed!" % (idx))


    def visualization(self, ep_num, batch_num):
        print("Time:", time.asctime( time.localtime(time.time()) )," EPOCH[%d]-Real_loss: %f Fake_loss: %f "
                        "Gen_loss: %f (Style_loss: %f, Grad_loss: %f, Targeted_loss: %f, TV loss: %f)" %
              (ep_num, self.disp_ad_true_loss/batch_num, self.disp_ad_fake_loss/batch_num, self.disp_g_loss/batch_num,
               self.style_l/batch_num, self.grad_l/batch_num, self.tar_l/batch_num, self.tv_l/batch_num))
        self.train_model_name_list = self.train_model_name_list[2:] + self.train_model_name_list[:2]
        torchvision.utils.save_image(self.orig_before[0:4] / 2 + 0.5, self.log_dir + '/before.png', nrow=1)
        torchvision.utils.save_image(self.orig_after[0:4] / 2 + 0.5, self.log_dir + '/after.png', nrow=1)
        torchvision.utils.save_image(self.gen_img[0:4] / 2 + 0.5, self.log_dir + '/res.png', nrow=1)
        torchvision.utils.save_image(self.orig_before_pasted[0:4] / 2 + 0.5,  
                                     self.log_dir + '/before_pasted.png', nrow=1)

