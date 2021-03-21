import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F
from glob import glob
import numpy as np
from PIL import Image
import kornia
import kornia.augmentation as K
from pathlib import Path

class CelebA(Dataset):
    """CelebA dataset + some extra stuff for downsampling, augmentations,
    transformations, and custom splits.

    Parameters
    ----------
    data_folder : str
        Path to CelebA dataset. This directory should contain two sub-
        directories, "images" and "labels", containing all images in
        the dataset and the list_attr_celeba.txt labels file,
        respectively
    fold : ["train", "val", "test", "full"], default="train"
        Fold to use. If "full", the entire dataset will be used.
    use_transforms : bool, default=False
        Toggles usage of data augmentations (horizontal flipping,
        random rotation, and random cropping)
    sample_rate : int, optional
        If provided, the dataset will be downsampled by sample_rate
        times. The randomly sampled subset will be cached to the file
        "samples/{sample_rate}.npy" to ensure future runs with that
        sample rate are using the same data.
    full_dir : str, optional
        If provided, should be a path to the uncropped CelebA images.
        The dataset will use these instead of the cropped/aligned
        images. Each image is 0-padded to be square then resized to
        500x500px.
    normalize : bool, default=False
        If true, images will be normalized using the ImageNet mean /
        std dev values.
    custom_splits : str, optional
        If provided, should be a path to a txt file in the same format
        as list_eval_partition.txt -- each line should contain an image
        file, followed by a space, followed by a 0 indicating train set
        and a 1 indicating validation set. Setting a custom test set
        is not supported (this is probably not something you should be
        doing anyway).
    """
    def __init__(self, data_folder, fold="train", use_transforms=False,
                 sample_rate=None, full_dir=None, normalize=False,
                 custom_splits=None):
        super().__init__()
        self.images = []
        self.labels = []
        self.fold = fold
        self.full_dir = full_dir
        if self.full_dir is not None:
            self.full_dir += "/"
        self.no_norm = not normalize

        self.create_folds(custom_splits)

        with open(data_folder + "/labels/list_attr_celeba.txt") as label_f:
            id_count = label_f.readline().strip("\n")
            self.attributes = label_f.readline().strip("\n").split()

            self.attributes = np.array(self.attributes)

            if sample_rate is not None:
                Path(f"samples").mkdir(exist_ok=True)
                if not Path(f"samples/{sample_rate}.npy").exists():
                    self.sampled_ids = np.random.choice(np.arange(162770),
                        162770//sample_rate, replace=False)
                    np.save(f"samples/{sample_rate}.npy", self.sampled_ids)
                else:
                    self.sampled_ids = np.load(f"samples/{sample_rate}.npy")

            for i, line in enumerate(label_f.read().split("\n")[:-1]):
                image_data = line.split()
                image = image_data[0]
                image_n = int(image[:-4])

                if sample_rate is not None:
                    if image_n not in self.sampled_ids:
                        continue

                image_labels = [int(label) for label in image_data[1:]]

                if self.fold_cond[fold](image_n):
                    if self.full_dir is None:
                        self.images.append(data_folder + "/images/" + image)
                    else:
                        self.images.append(self.full_dir + image)
                    self.labels.append(image_labels)

        self.length = len(self.images)

        self.use_transforms = use_transforms
        if self.use_transforms:
            self.flip_aug = K.RandomHorizontalFlip(p=0.5)
            self.rotation_aug = K.RandomRotation(degrees=5)
            if self.full_dir is not None:
                self.crop = K.RandomCrop((500,500), pad_if_needed=True)
            else:
                self.crop = K.RandomCrop((274,224), pad_if_needed=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def create_folds(self, custom_splits):
        """Helper function which creates a set of lambdas to determine
        which fold a given image is in. Can be configured by providing
        custom splits.
        """
        if custom_splits is not None:
            self.train_images = set()
            self.val_images = set()
            with open(custom_splits) as custom_splits_f:
                for line in custom_splits_f:
                    line_data = line.strip("\n").split()
                    image_n = int(line_data[0][:6])
                    if line_data[1] == "0":
                        self.train_images.add(image_n)
                    elif line_data[1] == "1":
                        self.val_images.add(image_n)
            self.fold_cond = {
                "train" : lambda image_n : image_n in self.train_images,
                "val" : lambda image_n : image_n in self.val_images,
                "test" : lambda image_n : image_n >= 182638,
                "full" : lambda image_n : True
            }
        else:
            self.fold_cond = {
                "train" : lambda image_n : image_n < 162771,
                "val" : lambda image_n : image_n >= 162771 and image_n<182638,
                "test" : lambda image_n : image_n >= 182638,
                "full" : lambda image_n : True
            }

    def __getitem__(self, index):
        image = TF.to_tensor(Image.open(self.images[index]))
        label = torch.tensor([1 if l == 1 else 0 for l in self.labels[index]])

        if self.full_dir is not None:
            input_shape = image.shape
            scale = 500/max(image.shape)
            size = (int(image.shape[1]*scale), int(image.shape[2]*scale))
        else:
            size = (274, 224)

        if self.use_transforms:
            image = image.unsqueeze(0)

            scale_aug = torch.rand(1).item()*.1+.95
            image = kornia.resize(image, (int(size[0]*scale_aug),
                                          int(size[1]*scale_aug)))
            image = self.crop(image)
            image = self.flip_aug(image)

            rot_params = self.rotation_aug.generate_parameters(image.shape)
            image = self.rotation_aug(image, rot_params).squeeze(0)

            image = image.squeeze(0)
        else:
            image = image.unsqueeze(0)
            image = kornia.resize(image, (size[0],size[1]))
            if self.full_dir is not None:
                pad_top = (500 - size[0])//2
                pad_bottom = (500 - size[0] + 1)//2
                pad_left = (500 - size[1])//2
                pad_right = (500 - size[1] + 1)//2
                image = F.pad(image,(pad_left, pad_right, pad_top, pad_bottom))
            image = image.squeeze(0)

        if not self.no_norm:
            return self.normalize(image), label
        else:
            return image, label

    def __len__(self):
        return self.length
