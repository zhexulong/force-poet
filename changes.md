# PoET
## ycbv2poet
- changed `annotation_path = 'train_pbr_only.json'` to `'train_pbr.json'` because PoET will look for `annotations/train_pbr.json`
  
## main.py
- needed to add `args.dataset` to `bop_evaluate` at line #330 due to exception
  
## pose_dataset.py
Refactored `PATHS` to ...

```
PATHS = {
    "train": (root, root / "annotations" / f'train.json'),
    "train_synt": (root, root / "annotations" / f'train_synt.json'),
    "train_pbr": (root, root / "annotations" / f'train_pbr.json'),
    "test": (root, root / "annotations" / f'test.json'),
    "test_all": (root, root / "annotations" / f'test.json'),
    "keyframes": (root, root / "annotations" / f'keyframes.json'),
    "keyframes_bop": (root, root / "annotations"/ f'keyframes_bop.json'),
    "val": (root / "val", root / "annotations" / f'val.json'),
}
```
... because `pose_dataset.py` will prepend e.g. `train/`, `test/`, etc., to the image file paths in e.g., `annotations/train.json`


## coco.py
- added `annFile = json.load(open(annFile, 'r'))` in constructor of `CocoDetection` because `annFile` is of type `PosixPath` and COCO requires eather `str` or `dict`

## model_tools.py
- changed used numpy dtype `float` to `np.float` due to exception

# Bop_Toolkit
## visualization.py
- changed `text_width, text_height = font.getsize(txt)` in line #70 to `text_width, text_height = font.getbbox(txt)[2:4]` because the former is deprecated
- changed `model_type` to `fine` because `eval` 3D models dont have a normal that is required for visualization???