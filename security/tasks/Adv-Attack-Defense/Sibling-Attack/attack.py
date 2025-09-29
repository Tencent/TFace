import os
import time
import torch
import torch.nn.functional as F
import models.models as m

from utils.utils import *
from omegaconf import OmegaConf


def load_surrogate_model():
    """ Load white-box and black-box models

    :return:
        face recognition and attribute recognition models
    """

    # Load pretrain white-box FR surrogate model
    fr_model = m.IR_152((112, 112))
    fr_model.load_state_dict(torch.load('./models/ir152.pth', weights_only=True))
    fr_model.to(device)
    fr_model.eval()

    # Load pretrain white-box AR surrogate model
    ar_model = m.IR_152_attr_all()
    ar_model.load_state_dict(torch.load('./models/ir152_ar.pth', weights_only=True))
    ar_model.to(device)
    ar_model.eval()

    return fr_model, ar_model

'''
    Obtain intermediate features by hooker
'''
layer_name = "ir_152.body.49"
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

gouts = []
def backward_hook(module, gin, gout):
    gouts.append(gout[0].data)
    return gin

def infer_fr_model(attack_img, victim_img, fr_model):
    """ Face recognition inference

    :param attack_img:
            attacker face image
    :param victim_img:
            victim face image
    :param fr_model:
            face recognition model
    :return:
        feature representations for the attacker and victim face images
    """
    attack_img_feat = fr_model(attack_img)
    victim_img_feat = fr_model(victim_img)
    return attack_img_feat, victim_img_feat

def infer_ar_model(attack_img, victim_img, ar_model):
    """ Face attribute recognition inference

    Args:
        :param attack_img:
            attacker face image
        :param victim_img:
            victim face image
        :param ar_model:
            attribute recognition model

    :return:
        intermediate feature representations for the attacker and victim face images
    """

    ar_model(attack_img)
    attack_img_mid_feat = activation[layer_name].clone()
    attack_img_mid_feat = torch.flatten(attack_img_mid_feat)
    attack_img_mid_feat = attack_img_mid_feat.expand(1, attack_img_mid_feat.shape[0])

    ar_model(victim_img)
    victim_img_mid_feat = activation[layer_name].clone().detach_()
    victim_img_mid_feat = torch.flatten(victim_img_mid_feat)
    victim_img_mid_feat = victim_img_mid_feat.expand(1, victim_img_mid_feat.shape[0])

    return attack_img_mid_feat, victim_img_mid_feat


def sibling_attack(attack_img, victim_img, fr_model, ar_model, config):
    """ Perform Sibling-Attack

    Args:
        :param attack_img:
            attacker face image
        :param victim_img:
            victim face image
        :param fr_model:
            face recognition model
        :param ar_model:
            attribute recognition model
        :param config:
            attacking configurations

    :return:
        adversarial face image
    """
    epochs = config.attack['outer_loops']
    alpha = config.attack['alpha']
    eps = config.attack['eps']
    INNER_MAX_EPOCH = config.attack['inner_loops']
    magic = config.attack['gamma']

    for layer in list(ar_model.named_modules()):
        if layer[0] == layer_name:
            fw_hook = layer[1].register_forward_hook(get_activation(layer_name))
            bw_hook = layer[1].register_backward_hook(backward_hook)

    ori_attack_img = attack_img.clone()
    for i in range(1, epochs+1):
        pre = time.time()
        if i % 2 == 0:
            INNER_LR = 1.0 / 255.0 * magic
            attack_img_tmp = attack_img.clone()
            attack_img_tmp_list = []

            for j in range(INNER_MAX_EPOCH):
                attack_img_tmp.requires_grad = True
                attack_img_feat, victim_img_feat = infer_fr_model(attack_img_tmp, victim_img, fr_model)
                fr_adv_loss = 1 - cos_simi(attack_img_feat, victim_img_feat)
                fr_model.zero_grad()
                fr_adv_loss.backward()
                sign_grad = attack_img_tmp.grad.sign()

                # core of PGD algorithm
                adv_img = attack_img_tmp - 1.0 * INNER_LR * sign_grad
                eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
                adv_img = torch.clamp(ori_attack_img + eta, min=-1, max=1).detach_()
                attack_img_tmp_list.append(adv_img)
                attack_img_tmp = adv_img.clone()

            while gouts:
                tensor = gouts.pop()
                tensor.detach_()

            AR_grad_temp_list = []
            for attack_img_tmp in attack_img_tmp_list:
                attack_img_tmp.requires_grad = True
                attack_img_mid_feat, victim_img_mid_feat = infer_ar_model(attack_img_tmp, victim_img, ar_model)
                ar_adv_loss = 1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)
                ar_model.zero_grad()

                ar_adv_loss.backward()
                grad = attack_img_tmp.grad
                AR_grad_temp_list.append(grad.clone())
                attack_img_tmp.detach_()

            aggr_grad_pic = torch.zeros_like(attack_img)
            for AR_grad_temp in AR_grad_temp_list:
                aggr_grad_pic += AR_grad_temp

            # use aggregrated gradients
            attack_img = attack_img_tmp_list[-1].clone()
            attack_img.requires_grad = True
            attack_img_mid_feat, victim_img_mid_feat = infer_ar_model(attack_img, victim_img, ar_model)
            ar_adv_loss = 1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)

            ar_model.zero_grad()
            ar_adv_loss.backward()

            w = 0.0001
            sign_grad = (attack_img.grad + w * aggr_grad_pic).sign()

            # core of PGD algorithm
            # use 1-step FR adv example as mid
            adv_img = attack_img - magic * alpha * sign_grad
            eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
            attack_img = torch.clamp(ori_attack_img + eta, min=-1, max=1).detach_()

            print("[Epoch-%d](FR-branch) AR loss: %f, time cost: %fs" % (i, ar_adv_loss.item(), time.time() - pre))
        else:
            INNER_LR = 1.0 / 255.0 * (1 - magic)
            attack_img_tmp = attack_img.clone()
            attack_img_tmp_list = []

            for j in range(INNER_MAX_EPOCH):
                attack_img_tmp.requires_grad = True
                attack_img_mid_feat, victim_img_mid_feat = infer_ar_model(attack_img_tmp, victim_img, ar_model)

                ar_adv_loss = 1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)
                ar_model.zero_grad()

                ar_adv_loss.backward()
                sign_grad = attack_img_tmp.grad.sign()

                # core of PGD algorithm
                adv_img = attack_img_tmp - 1.0 * INNER_LR * sign_grad
                eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
                adv_img = torch.clamp(ori_attack_img + eta, min=-1, max=1).detach_()
                attack_img_tmp_list.append(adv_img)
                attack_img_tmp = adv_img.clone()

            while gouts:
                tensor = gouts.pop()
                tensor.detach_()

            FR_grad_temp_list = []
            for attack_img_tmp in attack_img_tmp_list:
                attack_img_tmp.requires_grad = True
                attack_img_feat, victim_img_feat = infer_fr_model(attack_img_tmp, victim_img, fr_model)
                fr_adv_loss = 1 - cos_simi(attack_img_feat, victim_img_feat)
                fr_model.zero_grad()
                fr_adv_loss.backward()

                grad = attack_img_tmp.grad
                FR_grad_temp_list.append(grad.clone())
                attack_img_tmp.detach_()

            aggr_grad_pic = torch.zeros_like(attack_img)
            for FR_grad_temp in FR_grad_temp_list:
                aggr_grad_pic += FR_grad_temp

            # use aggregrated gradients
            attack_img = attack_img_tmp_list[-1].clone()
            attack_img.requires_grad = True
            attack_img_feat, victim_img_feat = infer_fr_model(attack_img, victim_img, fr_model)
            fr_adv_loss = 1 - cos_simi(attack_img_feat, victim_img_feat)

            fr_model.zero_grad()
            fr_adv_loss.backward()

            w = 0.0001
            sign_grad = (attack_img.grad + w * aggr_grad_pic).sign()

            # core of PGD algorithm
            # use 1-step FR adv example as mid
            adv_img = attack_img - (1 - magic) * alpha * sign_grad
            eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
            attack_img = torch.clamp(ori_attack_img + eta, min=-1, max=1).detach_()

            print("[Epoch-%d](AR-branch) FR loss: %f, time cost: %fs" % (i, fr_adv_loss.item(), time.time() - pre))

    return attack_img


if __name__ == '__main__':
    config = OmegaConf.load('./configs/config.yaml')
    gpu = config.attack['gpu']
    dataset_name = config.dataset['dataset_name']
    device = torch.device('cuda:' + str(gpu))

    fr_model, ar_model = load_surrogate_model()
    attack_img_paths, victim_img_paths = obtain_attacker_victim(config)
    pairs_num = len(attack_img_paths) * len(victim_img_paths)
    for attack_img_path in attack_img_paths:
        for victim_img_path in victim_img_paths:
            print(attack_img_path, "========", victim_img_path)

            attack_img = load_img(attack_img_path, config).to(device)
            victim_img = load_img(victim_img_path, config).to(device)

            # Perform Sibling-Attack
            adv_attack_img = sibling_attack(attack_img, victim_img, fr_model, ar_model, config)
            
            save_dir = './' + dataset_name +  '_results_adv_images/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = save_dir + victim_img_path.split('/')[-1].split('.')[0] + '+' +\
                        attack_img_path.split('/')[-1].split('.')[0] + '.png'
            save_adv_img(adv_attack_img.cpu(), save_path, config)
            print("Save adversarial image to - ", save_path)

