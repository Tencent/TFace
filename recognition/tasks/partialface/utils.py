import torch
import torch.fft
from torchjpeg import dct
from torch.nn import functional as F


def dct_transform(x, chs_remove=None, chs_pad=False,
                  size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
        Transform a spatial image into its frequency channels.
        Prune low-frequency channels if necessary.
    """

    # assert x is a (3, H, W) RGB image
    assert x.shape[1] == 3

    # convert the spatial image's range into [0, 1], recommended by TorchJPEG
    x = x * 0.5 + 0.5

    # up-sample
    x = F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)

    # convert to the YCbCr color domain, required by DCT
    x = x * 255
    x = dct.to_ycbcr(x)
    x = x - 128

    # perform block discrete cosine transform (BDCT)
    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)

    # prune channels
    if chs_remove is not None:
        channels = list(set([i for i in range(64)]) - set(chs_remove))
        if not chs_pad:
            # simply remove channels
            x_freq = x_freq[:, :, channels, :, :]
        else:
            # pad removed channels with zero, helpful for visualization
            x_freq[:, :, channels] = 0

    # stack frequency channels from each color domain
    x_freq = x_freq.reshape(b, -1, n_block, n_block)

    return x_freq


def idct_transform(x, size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
        The inverse of DCT transform.
        Transform frequency channels (must be 192 channels, can be padded with 0) back to the spatial image.
    """

    b, _, h, w = x.shape

    x = x.view(b, 3, 64, h, w)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(b, 3, h * w, 8, 8)
    x = dct.block_idct(x)
    x = x.view(b * 3, h * w, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(112 * ratio, 112 * ratio),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(b, 3, 112 * ratio, 112 * ratio)
    x = x + 128
    x = dct.to_rgb(x)
    x = x / 255
    x = F.interpolate(x, scale_factor=1 / ratio, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x


def create_channel_subsets(x, chs_prune=None, selection=None, n_subset=6):
    """
        Turn spatial images into subsets of frequency channels
    """

    # low-frequency channels to prune
    if chs_prune is None:
        chs_prune = [0, 1, 2, 3, 8, 9, 10, 16, 17, 24]

    # pre-specified combinations for subset selection (bold S in our paper)
    #   wlog., we split high-frequency channels into 6 subsets, each containing s=9 channels
    #   the specific combination can be randomly and model-wisely generated
    # trick: this implementation equals, in effect, drawing all subsets at the same time
    #   each of its row indicates a specific selection S
    #   only one subset is later chosen and permuted
    if selection is None:
        selection = torch.tensor([[1, 24, 28, 12, 32, 45, 47, 40, 52],
                                  [35, 39, 44, 42, 10, 4, 37, 31, 53],
                                  [0, 50, 19, 6, 22, 51, 49, 23, 21],
                                  [38, 8, 25, 13, 17, 30, 7, 33, 9],
                                  [26, 29, 16, 2, 43, 14, 5, 11, 48],
                                  [36, 41, 46, 34, 15, 27, 18, 20, 3]])
        assert selection.shape[0] == n_subset

    # turn the spatial image into frequency channels, where low-frequency channels are pruned
    x_freq = dct_transform(x, chs_remove=chs_prune, chs_pad=False)

    b, c, h, w = x_freq.shape
    x_freq = x_freq.reshape(b, 3, 64 - len(chs_prune), h, w)
    x_freq = torch.reshape(x_freq, (b, 3, c // 3, h, w))

    selection = selection.reshape(-1)

    # apply the combination to split frequency channels
    x_freq = x_freq[:, :, selection]
    x_freq = torch.reshape(x_freq, (b, 3, n_subset, c // (3 * n_subset), h, w))
    x_freq = x_freq.permute(2, 0, 1, 3, 4, 5)

    # dimension 0 is organized to be subset-first
    #   i.e., subset 0 of full batch of samples, then subset 1, etc.
    x_freq = torch.reshape(x_freq, (b * n_subset, c // n_subset, h, w))

    return x_freq


def form_training_batch(inputs, labels):
    """
        Create the training batch of random subsets of frequency channels
        from the standard inputs of spatial images
    """

    b, _, _, _ = inputs.shape

    # this specifies the image-wise potential choices of subsets based on the subset's index
    #   wlog., we select 3 out of 6 subsets for each sample image to form the training dataset
    #   this ensures a secure training, as the model cannot access all subsets of any given image
    #   len(choice_index) = C(3,6) = 20
    choice_index = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3],
                    [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [0, 4, 5],
                    [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
                    [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]

    # pre-specified combinations for subset permutation (bold P in our paper)
    candidate_permutations = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                           [7, 3, 0, 6, 5, 4, 1, 2, 8],
                                           [4, 8, 6, 0, 2, 5, 3, 7, 1]])

    # draw a random subset for each sample image
    # trick: the subsets each sample image is allowed to choose from is assigned by the image's *label index*
    #        (cannot use image index as it changes for each training epoch due to dataset shuffling)
    #   e.g., assume drawing a random subset for image ID 153752 with label ID 15232
    #         15232 % 20 = 12  =>  its allowed choices of subsets are choice_index[12], i.e., subsets 1, 2, 5
    #         then, one subset from 1, 2, 5 is selected
    label_idx_mod = [int(labels[i]) % len(choice_index) for i in range(b)]
    idx_within_choice = torch.randint(high=len(choice_index[0]), size=(b,)).tolist()
    split_idx = [i + b * choice_index[label_idx_mod[i]][idx_within_choice[i]] for i in range(b)]
    input_splits = create_channel_subsets(inputs, n_subset=6)[split_idx]

    # sample-wisely permute the order of channels
    ig_perm_idx = torch.randint(high=len(candidate_permutations), size=(input_splits.shape[0],))
    input_splits = input_splits.reshape(b, 3, 9, 112, 112)
    for i in range(b):
        input_splits[i] = input_splits[i, :, candidate_permutations[ig_perm_idx[i]]]
    inputs = input_splits.reshape(b, -1, 112, 112)

    return inputs, labels


if __name__ == '__main__':
    x = torch.rand(16, 3, 112, 112)
    y = torch.randint(10000, (16,))

    x_freq = dct_transform(x)
    x_spat = idct_transform(x_freq)
    print(x_freq.shape, x_spat.shape)  # torch.Size([16, 192, 112, 112]) torch.Size([16, 3, 112, 112])

    print(create_channel_subsets(x).shape)  # torch.Size([96, 27, 112, 112])

    print(form_training_batch(x, y)[0].shape)  # torch.Size([16, 27, 112, 112])
