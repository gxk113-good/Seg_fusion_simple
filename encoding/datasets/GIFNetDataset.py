import os
import random
import torch
import pickle
import numpy as np
from tqdm import tqdm
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps, ImageFilter
from .base import BaseDataset

class CustomDataset(BaseDataset):
    NUM_CLASS = 9  # 替换为你的类别数
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CustomDataset, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        self.ir_images,self.vis_images, self.masks = _get_custom_pairs(root, split)
        assert (len( self.vis_images) == len(self.masks))
        # if len(self.ir_images) == 0:
        #     raise(RuntimeError("Found 0 images in subfolders of: \
        #         " + root + "\n"))
        # self.root = root


# 添加以下两行初始化 _indices 和 _classes
        self._indices = list(range(self.NUM_CLASS))  # 创建类别索引 [0, 1, 2, ..., 8]
        self._classes = np.array(self._indices)     # 创建对应的类别数组
        
    @property
    def pred_offset(self):
        return 0  # 根据实际数据集需求返回偏移量

        # self.image_numbers = image_numbers
        # self.transform = transform
        # self.target_transform = target_transform

    # def __len__(self):
    #     return len(self.image_numbers)

    def __getitem__(self, index):
        # ir_path = os.path.join(self.root, "infrared/train/", f"{idx + 1}.jpg")
        # vis_path = os.path.join(self.root, "visible/train/", f"{idx + 1}.jpg")
        # visNF_path = os.path.join(self.root, "visible_focus_near/train/", f"{idx + 1}.jpg")
        # visFF_path = os.path.join(self.root, "visible_focus_far/train/", f"{idx + 1}.jpg")
        # mask_path = os.path.join(self.root, "masks/train/", f"{idx + 1}.png")
        # ir_img = Image.open(ir_path).convert("L")
        # vis_img = Image.open(vis_path).convert("L")
        # mask = Image.open(mask_path)
        ir_img = Image.open(self.ir_images[index]).convert('RGB')
        vis_img = Image.open(self.vis_images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        if self.mode == 'test':
            if self.transform is not None:
                ir_img = self.transform(ir_img)
                vis_img = self.transform(vis_img)
            mask = self._mask_transform(mask) 
            return ir_img, vis_img,mask   #os.path.basename(self.ir_images[index])  #这里为什么只有ir
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            ir_img,vis_img, mask = self._sync_transform(ir_img, vis_img, mask)
        elif self.mode == 'val':
            ir_img,vis_img, mask = self._val_sync_transform(ir_img, vis_img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
             ir_img = self.transform(ir_img)
             vis_img = self.transform(vis_img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return ir_img, vis_img, mask
    
    def _sync_transform(self, ir_img,vis_img, mask):
        # random mirror
        if random.random() < 0.5:
            ir_img = ir_img.transpose(Image.FLIP_LEFT_RIGHT)
            vis_img=vis_img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = ir_img.size
        
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        ir_img = ir_img.resize((ow, oh), Image.BILINEAR)
        vis_img = vis_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        ir_img = ir_img.rotate(deg, resample=Image.BILINEAR)
        vis_img = vis_img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            ir_img = ImageOps.expand(ir_img, border=(0, 0, padw, padh), fill=0)
            vis_img = ImageOps.expand(vis_img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = ir_img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        ir_img = ir_img.crop((x1, y1, x1+crop_size, y1+crop_size))
        vis_img = vis_img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            ir_img = ir_img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            vis_img = vis_img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        return ir_img,vis_img, self._mask_transform(mask)

        
        # visNF_img = Image.open(visNF_path).convert("L")
        # visFF_img = Image.open(visFF_path).convert("L")

        # if self.transform:
        #     ir_img = self.transform(ir_img)
        #     vis_img = self.transform(vis_img)
            # visNF_img = self.transform(visNF_img)
            # visFF_img = self.transform(visFF_img)

        # return ir_img, vis_img                    #, visNF_img, visFF_img

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64')  # 根据需要调整数据类型
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.ir_images)    #为什么只有ir
    
    def make_pred(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._indices)
        index = np.digitize(mask.ravel(), self._indices, right=True)
        return self._classes[index].reshape(mask.shape)

# def _get_custom_pairs(folder, split='train'):
#     # 实现获取图像和掩码路径的逻辑
#     # 这里需要根据你的数据集组织方式来实现
#     ir_img_paths = []
#     vis_img_paths = []
#     mask_paths = []
#     if split == 'train':
#         ir_img_folder = os.path.join(folder, 'infrared/train')
#         vis_img_folder = os.path.join(folder, 'visible/train')
#         mask_folder = os.path.join(folder, 'masks/train')
#         # 遍历文件夹获取图像和掩码路径
#         for filename in os.listdir(ir_img_folder):
#             if filename.endswith(".png"):  # 根据你的图像格式调整
#                 ir_img_path = os.path.join(ir_img_folder, filename)
#                 vis_img_path = os.path.join(vis_img_folder, filename)
#                 mask_filename = filename.replace('.jpg','.png')  # 根据你的掩码格式调整
#                 mask_path = os.path.join(mask_folder, mask_filename)
#                 if os.path.exists(mask_path) and os.path.exists(vis_img_path):
#                     ir_img_paths.append(ir_img_path)
#                     vis_img_paths.append(vis_img_path)
#                     mask_paths.append(mask_path)
#     # 实现其他 split 的逻辑...
#     return ir_img_paths, vis_img_paths, mask_paths

def _get_custom_pairs(folder, split='train'):
    # 实现获取图像和掩码路径的逻辑
    # 这里需要根据你的数据集组织方式来实现
    ir_img_paths = []
    vis_img_paths = []
    mask_paths = []
    
    if split == 'train':
        ir_img_folder = os.path.join(folder, 'infrared/train')
        vis_img_folder = os.path.join(folder, 'visible/train')
        mask_folder = os.path.join(folder, 'masks/train')
        # 遍历文件夹获取图像和掩码路径
        for filename in os.listdir(ir_img_folder):
            if filename.endswith(".png"):  # 根据你的图像格式调整
                ir_img_path = os.path.join(ir_img_folder, filename)
                vis_img_path = os.path.join(vis_img_folder, filename)
                mask_filename = filename.replace('.jpg', '.png')  # 根据你的掩码格式调整
                mask_path = os.path.join(mask_folder, mask_filename)
                if os.path.exists(mask_path) and os.path.exists(vis_img_path):
                    ir_img_paths.append(ir_img_path)
                    vis_img_paths.append(vis_img_path)
                    mask_paths.append(mask_path)
    elif split == 'val':
        ir_img_folder = os.path.join(folder, 'infrared/val')
        vis_img_folder = os.path.join(folder, 'visible/val')
        mask_folder = os.path.join(folder, 'masks/val')
        # 遍历文件夹获取图像和掩码路径
        for filename in os.listdir(ir_img_folder):
            if filename.endswith(".png"):  # 根据你的图像格式调整
                ir_img_path = os.path.join(ir_img_folder, filename)
                vis_img_path = os.path.join(vis_img_folder, filename)
                mask_filename = filename.replace('.jpg', '.png')  # 根据你的掩码格式调整
                mask_path = os.path.join(mask_folder, mask_filename)
                if os.path.exists(mask_path) and os.path.exists(vis_img_path):
                    ir_img_paths.append(ir_img_path)
                    vis_img_paths.append(vis_img_path)
                    mask_paths.append(mask_path)
    elif split == 'test':
        ir_img_folder = os.path.join(folder, 'infrared/test')
        vis_img_folder = os.path.join(folder, 'visible/test')
        mask_folder = os.path.join(folder, 'masks/test')
        # 遍历文件夹获取图像和掩码路径
        for filename in os.listdir(ir_img_folder):
            if filename.endswith(".png"):  # 根据你的图像格式调整
                ir_img_path = os.path.join(ir_img_folder, filename)
                vis_img_path = os.path.join(vis_img_folder, filename)
                mask_filename = filename.replace('.jpg', '.png')  # 根据你的掩码格式调整
                mask_path = os.path.join(mask_folder, mask_filename)
                if os.path.exists(mask_path) and os.path.exists(vis_img_path):
                    ir_img_paths.append(ir_img_path)
                    vis_img_paths.append(vis_img_path)
                    mask_paths.append(mask_path)
    # 可以继续添加其他 split 的逻辑，例如 'test'
    else:
        raise ValueError(f"Unsupported split: {split}")
    
    return ir_img_paths, vis_img_paths, mask_paths
    
   