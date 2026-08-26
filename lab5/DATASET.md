# Dataset: Oxford-IIIT Pet Dataset

**Source:** https://www.robots.ox.ac.uk/~vgg/data/pets/

## Description

| Property | Value |
|---|---|
| Instances | 7,349 images (3,680 trainval, 3,669 test) |
| Classes | 37 pet breeds (cats and dogs) |
| Format | RGB JPEG, variable native resolution |
| Preprocessing used | Resized to 224 x 224 x 3, scaled to [-1, 1] (MobileNetV2 convention) |
| Task | Multi-class image classification |
| Annotations | Class ID, species (cat/dog), breed ID -- per `annotations/trainval.txt` and `annotations/test.txt` |

## Why This Dataset

Oxford-IIIT Pet is small enough to train quickly (even on a single Colab GPU
or CPU) while still being a genuine multi-class, moderately fine-grained
classification problem -- well suited to demonstrating transfer learning with
an ImageNet-pretrained backbone such as MobileNetV2.

## Download

The dataset is **not** committed to this repository (see `.gitignore`) due to
its size (~800 MB). All provided scripts download and extract it
automatically on first run:

```
https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

To download manually instead:

```bash
mkdir -p pets && cd pets
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

This produces:

```
pets/
├── images/            -- all JPEGs, named <Breed>_<index>.jpg
└── annotations/
    ├── trainval.txt    -- training/validation split: "<name> <class_id> <species_id> <breed_id>"
    ├── test.txt         -- test split, same format
    ├── list.txt          -- full annotation list
    ├── trimaps/            -- pixel-level segmentation masks (unused in this experiment)
    └── xmls/                -- bounding-box annotations (unused in this experiment)
