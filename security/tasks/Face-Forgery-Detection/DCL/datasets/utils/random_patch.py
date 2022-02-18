import cv2
import numpy as np
import torch


PERMUTATIONS_40 = [
    [7, 2, 0, 1, 5, 6, 3, 4, 8],
    [0, 1, 2, 3, 4, 5, 6, 8, 7],
    [1, 0, 3, 2, 6, 4, 8, 7, 5],
    [2, 3, 1, 0, 7, 8, 4, 5, 6],
    [3, 4, 5, 6, 8, 0, 7, 1, 2],
    [4, 5, 6, 8, 0, 7, 1, 2, 3],
    [5, 6, 8, 7, 1, 2, 0, 3, 4],
    [6, 8, 7, 4, 2, 3, 5, 0, 1],
    [8, 7, 4, 5, 3, 1, 2, 6, 0],
    [0, 1, 2, 3, 8, 7, 5, 6, 4],
    [1, 0, 4, 6, 2, 5, 8, 7, 3],
    [2, 3, 5, 4, 7, 1, 0, 8, 6],
    [3, 2, 6, 5, 1, 4, 7, 0, 8],
    [4, 5, 8, 7, 0, 6, 2, 3, 1],
    [5, 4, 7, 1, 6, 8, 3, 2, 0],
    [8, 6, 0, 2, 4, 3, 1, 5, 7],
    [6, 7, 3, 8, 5, 0, 4, 1, 2],
    [7, 8, 1, 0, 3, 2, 6, 4, 5],
    [0, 1, 2, 4, 3, 7, 6, 5, 8],
    [1, 0, 3, 2, 4, 6, 7, 8, 5],
    [2, 3, 0, 1, 5, 4, 8, 6, 7],
    [3, 2, 1, 0, 6, 8, 5, 7, 4],
    [4, 5, 6, 7, 8, 3, 1, 2, 0],
    [5, 4, 8, 3, 7, 0, 2, 1, 6],
    [6, 8, 7, 5, 0, 1, 3, 4, 2],
    [7, 6, 5, 8, 1, 2, 4, 0, 3],
    [8, 7, 4, 6, 2, 5, 0, 3, 1],
    [0, 1, 2, 5, 4, 7, 8, 3, 6],
    [1, 0, 3, 2, 5, 8, 6, 4, 7],
    [2, 3, 0, 1, 6, 5, 4, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
]


class RandomPatch:
    def __init__(self, grid_size=3):
        """Random shuffle patches of images

        Args:
            grid_size (int, optional):  Defaults to 3.
        """
        self.grid_size = grid_size
        self.n_grids = grid_size ** 2
        self.permutations = PERMUTATIONS_40

    def __call__(self, image, mask=None):
        img_tiles = [None] * self.n_grids
        mask_tiles = [None] * self.n_grids

        for n in range(self.n_grids):
            img_tiles[n] = self._get_tile(image, n)
            if mask != None:
                mask_tiles[n] = self._get_tile(mask, n)
        order = np.random.randint(len(self.permutations))
        img_data = [img_tiles[self.permutations[order][t]]
                    for t in range(self.n_grids)]
        img_data = torch.stack(img_data, 0)
        image = self._stack_together(image, img_data)
        if mask != None:
            mask_data = [mask_tiles[self.permutations[order][t]]
                         for t in range(self.n_grids)]
            mask_data = torch.stack(mask_data, 0)
            mask = self._stack_together(mask, mask_data)
            return image, mask
        return image

    def _get_tile(self, img, n):
        """

        Args:
            img ([type])
            n ([type])

        Returns:
            [type]
        """
        w = int(img.size(1) / self.grid_size)
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img[:, x*w: (x+1)*w, y*w:(y+1)*w]
        return tile

    def _stack_together(self, img, img_tiles):
        """Stack all patches together

        Args:
            img ([tensor]):
            img_tiles ([tensor])

        Returns:
            [tensor]
        """
        num = img_tiles.size(0)
        subwidth = img_tiles.size(-1)
        originalpic = torch.zeros_like(img)
        originalpic[:, :, :] = img[:, :, :]
        for i in range(num):
            y = int(i / self.grid_size)
            x = int(i % self.grid_size)
            originalpic[:, x*subwidth:(x+1)*subwidth, y *
                        subwidth:(y+1)*subwidth] = img_tiles[i, :, :, :]
        return originalpic


if __name__ == '__main__':
    rp = RandomPatch(grid_size=3)

    image_path = '../faces/0000.png'
    img = cv2.imread(image_path)
    img = cv2.resize(img, (320, 320))
    cv2.imwrite('img.png', img)

    img2 = torch.from_numpy(img.copy()).permute(2, 0, 1)
    img_rp = rp(img2).permute(1, 2, 0)
    cv2.imwrite('img_rp.png', img_rp.numpy())
