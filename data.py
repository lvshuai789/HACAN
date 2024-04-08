from faulthandler import is_enabled
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod

from transformers import BertTokenizer
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler

import six

def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:

            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()

    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3.txt?")

def convert_to_feature(raw, seq_length, tokenizer):
    line = convert_to_unicode(raw)
    tokens_a = tokenizer.tokenize(line)

    #print(tokens_a)
    #assert 1==0
    # Modifies `tokens_a` in place so that the total
    # length is less than the specified length.
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length
    return tokens, input_ids, input_mask, input_type_ids


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))[:25000]
        #ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))[20000:25000]
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, max_len, tokenizer, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        #
        self.max_len = max_len
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        #

        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root, caption, img_id, path, image, ann_id = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)
        

        # Convert caption (string) to word ids.
        is_bert = True
        if is_bert:
            tokens, input_ids, input_mask, input_type_ids = convert_to_feature(caption, self.max_len, self.tokenizer)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.long)
            input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
            # return image, input_ids, index, img_id, input_mask, input_type_ids
            return image, input_ids, index, img_id, input_mask, input_type_ids, ann_id
        else:
            tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            return image, target, index, img_id


        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')

        return root, caption, img_id, path, image, ann_id

    def __len__(self):
        return len(self.ids)


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, max_len, tokenizer, transform=None):

        #
        self.max_len = max_len
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        #

        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]
        # print(len(self.ids))

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        is_bert = True
        if is_bert:
            tokens, input_ids, input_mask, input_type_ids = convert_to_feature(caption, self.max_len, self.tokenizer)
            # with open("/home/ls/case/f30k_val_i2t_tokens.txt", "a") as f:
            #     f.write(str(caption) + "@" + str(tokens) + "\n")
            # if index >= len(self.ids):
            #     assert 0 ==1
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.long)
            input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
            return image, input_ids, index, img_id, input_mask, input_type_ids, ann_id[1]
        else:
            tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            return image, target, index, img_id  

    def __len__(self):
        return len(self.ids)
        # return 5000



def collate_fn_bert(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: torch.sum(x[-2]), reverse=True)
    images, input_ids, ids, img_ids, input_mask, input_type_ids, ann_id = zip(*data)

    # Merge images (convert tuple of 2D tensor to DD tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [torch.sum(cap) for cap in input_mask]
    input_ids = torch.stack(input_ids, 0)
    input_mask = torch.stack(input_mask, 0)
    input_type_ids = torch.stack(input_type_ids, 0)
    ids = np.array(ids)
    return images, input_ids, lengths, ids, input_mask, input_type_ids, img_ids, ann_id

def get_transform(data_name, split_name, crop_size):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


class F30kDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()

        #_config['data_root'], _config['datasets'], vocab, _config['max_text_len'], _config['tokenizer'], _config['image_size'], _config['per_gpu_batchsize'], _config['num_workers'])

        self.data_path = os.path.join(_config["data_root"], _config['datasets'])
        self.datasets = _config['datasets']
        self.vocab = None

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

        self.tokenizer = _config["tokenizer"]

        self.setup_flag = False
        self.dist = dist

        self.roots, self.ids = get_paths(self.data_path, self.datasets)

    def set_train_dataset(self):
        transform = get_transform(self.datasets, 'train', self.image_size)

        self.train_dataset = FlickrDataset(root=self.roots['train']['img'],
                                split='train',
                                json=self.roots['train']['cap'],
                                vocab=self.vocab,
                                max_len=self.max_text_len,
                                tokenizer=self.tokenizer,
                                transform=transform)

    def set_val_dataset(self):
        transform = get_transform(self.datasets, 'val', self.image_size)

        self.val_dataset = FlickrDataset(root=self.roots['val']['img'],
                                split='val',
                                json=self.roots['val']['cap'],
                                vocab=self.vocab,
                                max_len=self.max_text_len,
                                tokenizer=self.tokenizer,
                                transform=transform)

    def set_test_dataset(self):
        transform = get_transform(self.datasets, 'test', self.image_size)

        self.test_dataset = FlickrDataset(root=self.roots['test']['img'],
                                split='test',
                                json=self.roots['test']['cap'],
                                vocab=self.vocab,
                                max_len=self.max_text_len,
                                tokenizer=self.tokenizer,
                                transform=transform)

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True

        '''if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None'''

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                              batch_size=self.batch_size,
                                              #sampler=self.train_sampler,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=self.num_workers,
                                              collate_fn=collate_fn_bert)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                              batch_size=self.batch_size,
                                              #sampler=self.val_sampler,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=self.num_workers,
                                              collate_fn=collate_fn_bert)
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                              batch_size=self.batch_size,
                                              #sampler=self.val_sampler,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=self.num_workers,
                                              collate_fn=collate_fn_bert)
        return loader



class MscocoDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()

        #_config['data_root'], _config['datasets'], vocab, _config['max_text_len'], _config['tokenizer'], _config['image_size'], _config['per_gpu_batchsize'], _config['num_workers'])

        self.data_path = os.path.join(_config["data_root"], _config['datasets'])
        self.datasets = _config['datasets']
        self.vocab = None

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

        self.tokenizer = _config["tokenizer"]

        self.setup_flag = False
        self.dist = dist

        self.roots, self.ids = get_paths(self.data_path, self.datasets)

    def set_train_dataset(self):
        transform = get_transform(self.datasets, 'train', self.image_size)
    
        self.train_dataset = CocoDataset(root=self.roots['train']['img'],
                                json=self.roots['train']['cap'],
                                vocab=self.vocab,
                                max_len=self.max_text_len,
                                tokenizer=self.tokenizer,
                                transform=transform, ids=self.ids['train'])

    def set_val_dataset(self):
        transform = get_transform(self.datasets, 'val', self.image_size)

        self.val_dataset = CocoDataset(root=self.roots['val']['img'],
                                json=self.roots['val']['cap'],
                                vocab=self.vocab,
                                max_len=self.max_text_len,
                                tokenizer=self.tokenizer,
                                transform=transform, ids=self.ids['val'])

    def set_test_dataset(self):
        transform = get_transform(self.datasets, 'test', self.image_size)

        self.test_dataset = CocoDataset(root=self.roots['test']['img'],
                                json=self.roots['test']['cap'],
                                vocab=self.vocab,
                                max_len=self.max_text_len,
                                tokenizer=self.tokenizer,
                                transform=transform, ids=self.ids['test'])

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True

        '''if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None'''

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                              batch_size=self.batch_size,
                                              #sampler=self.train_sampler,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=self.num_workers,
                                              collate_fn=collate_fn_bert)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                              batch_size=self.batch_size,
                                              #sampler=self.val_sampler,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=self.num_workers,
                                              collate_fn=collate_fn_bert)
        return loader


    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                              batch_size=self.batch_size,
                                              #sampler=self.val_sampler,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=self.num_workers,
                                              collate_fn=collate_fn_bert)
        return loader


