
## Data Augmentation

We implement several data augmentation methods, include traditional image augmentation methods, such like `ShearXY，TranslateXY，Rotate，AutoContrast，Invert，Equalize，Solarize，Posterize，Contrast，Color，Brightness，Sharpness，Cutout`, and `Addmask，Addglass` to generate face images wearing glasses and masks.


### Glass and mask
The glass style is randomly chosed from four templates.

The mask style is randomly chosed from five templates and the `scale` controls the cover magnitude.

### Headscarf
`landmark_AddHeadband2Face.py` is the code for generating face images wearing fake headscarves, the input is the image list and corresponding 68 points landamark list. 
