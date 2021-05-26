
## Data Augmentation

### Introduction
We implement several data augmentation methods, include traditional image augmentation methods, such like `ShearXY，TranslateXY，Rotate，AutoContrast，Invert，Equalize，Solarize，Posterize，Contrast，Color，Brightness，Sharpness，Cutout`, and `Addmask，Addglass` to generate face images wearing glasses and masks.


### Glass and mask
Below images are the fake glass demo, the glass style is randomly chosed from four templates.
<p align="center">
  <img src="doc/yfliu_glass.png" title="glass" />
</p>

Below images are the fake mask demo, the mask style is randomly chosed from five templates and the `scale` controls the cover magnitude.
<p align="center">
  <img src="doc/yfliu_mask.png" title="mask" />
</p>

### Headscarf
`landmark_AddHeadband2Face.py` is the code for generating face images wearing fake headscarves, the input is the image list and corresponding 68 points landamark list. Below images are the several headscarf styles.
<p align="center">
  <img src="doc/yfliu_headband.png" title="headband" />
</p>
