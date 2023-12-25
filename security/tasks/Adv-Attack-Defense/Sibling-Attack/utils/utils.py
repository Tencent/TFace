import cv2
import torch
import torch.nn.functional as F

def load_img(img_path, config):
    """ Read image from path and convert it to torch tensor

    Args:
        :param img_path:
            the path to the input face image
        :param config:
            attacking configurations

    :return:
        the processed face image torch tensor object
    """
    m = config.dataset['mean']
    s = config.dataset['std']
    input_size = tuple(config.dataset['input_size'])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32)
    img = img.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    mean = torch.tensor(m).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(s).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    img = (img - mean) / std
    img = F.interpolate(img, size=input_size, mode='bilinear')
    return img

def save_adv_img(attack_img, save_path, config):
    """ Save adversarial image to the distination path

    Args:
        :param attack_img:
            adversarial face image
        :param save_path:
            the path to save the adversarial face image
        :param config:
            attacking configurations

    :return:
        None
    """
    input_size = tuple(config.dataset['input_size'])
    m = config.dataset['mean']
    s = config.dataset['std']
    mean = torch.tensor(m).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(s).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    attack_img = F.interpolate(attack_img, size=input_size, mode='bilinear')
    attack_img = (attack_img * mean + std) * 255
    attack_img = attack_img[0].transpose(0, 1).transpose(1, 2)
    attack_img = attack_img.numpy()
    attack_img = cv2.cvtColor(attack_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, attack_img)

def cos_simi(emb_attack_img, emb_victim_img):
    """ Calculate cosine similarity between two face features

    Args:
        :param emb_attack_img:
            input feature representation for the attacker
        :param emb_victim_img:
            input feature representation for the victim

    :return:
        the cosine similarity of two features
    """
    return torch.mean(torch.sum(torch.mul(emb_victim_img, emb_attack_img), dim=1)
                      / emb_victim_img.norm(dim=1) / emb_attack_img.norm(dim=1))

def obtain_attacker_victim(config):
    """ Obtain attackers and victims' image paths

    Args:
        :param config:
            attacking configurations

    :return:
        the split path groups of attack and victim face images
    """
    dataset_dir = config.dataset['dataset_dir']
    dataset_txt = config.dataset['dataset_txt']
    attack_img_paths = []
    victim_img_paths = []
    with open(dataset_txt) as fin:
        img_names = fin.readlines()
        for idx, img_name in enumerate(img_names):
            img_path = dataset_dir + '/' + img_name.strip()
            if idx < 10:
                victim_img_paths.append(img_path)
            else:
                attack_img_paths.append(img_path)

    return attack_img_paths, victim_img_paths

