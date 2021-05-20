from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def gentxt(data_root, outfile):         # generate data file via traveling the target dataset 
    '''
    Use ImageFolder method to travel the target dataset 
    Save to two files including ".label" and ".labelpath"
    '''
    # output file1
    outfile1 = open(outfile, 'w')
    # output file2
    outfile2 = open(outfile+'path', 'w') 
    data = ImageFolder(data_root)
    count = 0
    tqdm(data.imgs)
    # travel the target dataset
    for index, value in enumerate(data.imgs):
        count += 1
        img = '/'+'/'.join(value[0].split('/')[-2:])
        outfile1.write(img + '\n')
        outfile2.write(value[0] + '\t' + str(value[1]) + '\n')
    return count
    
if __name__ == "__main__":                  # obtain data list
    '''
    This method is to obtain data list from dataset
    and save to txt files
    '''
    outfile = './DATA.label'
    data_root = '/data/home/fuzhaoou/Dataset/MS1M/MS1M_arcface_112'
    print(gentxt(data_root, outfile))
