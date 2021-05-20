import os
import numpy as np
import pandas as pd
import csv
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional

import configuration as cfg

# Returns an image record, its index, and its label.
class OpenImageVRD(Dataset):

    def __init__(self, root_dir, route, data_folder, data_csv, fraction):

        self.relationship_dict = {'at': 0, 'on': 1, 'holds': 2, 'plays': 3, 'interacts_with': 4, 'wears': 5, 'inside_of': 6, 'under': 7, 'hits': 8, 'is': 9}
        self.root_dir = root_dir
        self.vrd_csv = os.path.join(root_dir, data_csv)
        self.class_csv = os.path.join(root_dir, 'challenge-2019-classes-vrd.csv')
        self.attr_csv = os.path.join(root_dir, 'challenge-2019-attributes-description.csv')
        self.image_root = os.path.join(root_dir, data_folder)

        self.class_dict = self.getDict([self.class_csv, self.attr_csv]) # returns a dictionary.
        self.data_record = self.read_csv() # returns a list.

        self.resize_height = 224
        self.resize_width = 224
        self.crop_size = 112
        self.route = route # Determine whether to just crop, or concatenate the image with mask(s). Determined in configuration.py.
        self.fraction = fraction # dataset usage. Determined in configuration.py.

    # determining the amount of dataset to train and validate. 
    # returning the size of the dataset.
    def __len__(self):
    
        return int(len(self.data_record)*self.fraction) 

    # returns an image in Tensor format and its label.
    def __getitem__(self, index):
        row = self.data_record[index]
        img_id = row[0]
        bboxs = row[3:11]
        relation_id = row[-1]
        label = np.array(self.relationship_dict[relation_id], dtype=int)
        img_path = os.path.join(self.image_root, img_id + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # first option is crop.
        # second is concatenation of image and 2 masks resutling in 5-channel image.
        # third is a 4 channel image by concatenating a "overall mask" with the image.
        # last is just image resize to 224 x 224.
        if self.route == 0:
            img = self.getCropImage(img, bboxs, height, width)[0]
        elif self.route == 1:
            box = self.getBboxs(bboxs, height, width)
            masks1 = self.createMask(box[0], height, width)
            masks2 = self.createMask(box[1], height, width)
            img = self.image_resize_inception(img)
            img = np.concatenate([img, masks1, masks2], axis=2)
        elif self.route == 2:
            box = self.getCropImage(img, bboxs, height, width)
            xmin, ymin, xmax, ymax = box[1:]
            bbox = [xmin, ymin, xmax, ymax]
            mask = self.createMask(bbox, height, width)
            img = self.image_resize_inception(img)
            img = np.concatenate([img, mask], axis=2)
        else:
            img = self.image_resize(img)

        img = functional.to_tensor(img)
        if self.route == 0:
            img = functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, label 

    # returns bounding box coordinates for that particular image.
    def getBboxs(self, x, h, w):
        bbox1 = [int(float(x[0]) * w), int(float(x[2]) * h), int(float(x[1]) * w), int(float(x[3]) * h)]
        bbox2 = [int(float(x[4]) * w), int(float(x[6]) * h), int(float(x[5]) * w), int(float(x[7]) * h)]

        return [bbox1, bbox2]

    # creates a mask and returns it.
    def createMask(self, bbox, h, w):
        bbox = [int(x) for x in bbox]
        mask = np.zeros((h, w))
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        mask = self.image_resize_inception(mask)
        mask = np.expand_dims(mask, axis=2)

        return mask

    # source: https://stackoverflow.com/questions/59272589/get-object-from-bounding-box-object-detection
    def getCropImage(self, img, x, h, w):
        xmin1, ymin1, xmax1, ymax1 = [int(float(x[0]) * w), int(float(x[2]) * h), int(float(x[1]) * w),
                                      int(float(x[3]) * h)]
        xmin2, ymin2, xmax2, ymax2 = [int(float(x[4]) * w), int(float(x[6]) * h), int(float(x[5]) * w),
                                      int(float(x[7]) * h)]

        xmin = min(xmin1, xmax1, xmin2, xmax2)
        xmax = max(xmin1, xmax1, xmin2, xmax2)
        ymin = min(ymin1, ymax1, ymin2, ymax2)
        ymax = max(ymin1, ymax1, ymin2, ymax2)

        crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        crop_img = self.image_resize(crop_img)

        return [crop_img, xmin, ymin, xmax, ymax]

    # image resize is 224 x 224.
    def image_resize(self, img):
        resized_img = cv2.resize(img, dsize=(self.resize_width, self.resize_height), interpolation=cv2.INTER_NEAREST)

        return resized_img

    # image resize is 299 x 299.
    def image_resize_inception(self, img):
        resized_img = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_NEAREST)

        return resized_img

    # resource: https://stackoverflow.com/questions/209840/convert-two-lists-into-a-dictionary
    def getDict(self, filename):
        if isinstance(filename, str):
            file = pd.read_csv(filename, header=None)
        elif isinstance(filename, list):
            df = []
            for file in filename:
                file = pd.read_csv(file, header=None)
                df.append(file)
            file = pd.concat(df, ignore_index=True)

        file_dict = dict(zip(list(file.iloc[:, 0]), list(file.iloc[:, 1])))
        # file_list = list(file.iloc[:, 0])

        return file_dict

    # resource: https://stackoverflow.com/questions/24662571/python-import-csv-to-list
    def read_csv(self):
        with open(self.vrd_csv, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            # data_header = data[:1]
            data_records = data[1:]
            
            return data_records


if __name__ == '__main__':
    root_dir = r'C:\Users\anant\Downloads\OpenImages\dummy_data'
    data_train_folder = 'train'
    data_val_folder = 'validation'
    data_train_csv = 'challenge-2019-train-vrd.csv'
    data_val_csv = 'challenge-2019-validation-vrd.csv'
    train_set = OpenImageVRD(root_dir, cfg.route[2], data_train_folder, data_train_csv, cfg.fraction)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)
    for i, (img, label) in enumerate(train_loader):
        print(img.shape)
        print(label)
        print('=' * 20)
