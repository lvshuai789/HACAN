import torch
import numpy as np
import torch.nn as nn
from .bert import BertModel
import pytorch_lightning as pl
import torch.nn.functional as F
from . import objectives, meter_utils
import meter.modules.convnext as convnext
from meter.modules.visual_encoder import Visual_Enconder
from meter.modules.textual_encoder import Textual_Enconder


def freeze_layers(model, bool):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = bool


class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.eval_bool = False

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        #
        self.cross_modal_text_transform_1 = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform_1.apply(objectives.init_weights)
        self.cross_modal_text_transform_2 = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform_2.apply(objectives.init_weights)

        self.cross_modal_image_transform_1 = nn.Linear(256, config['hidden_size'])
        self.cross_modal_image_transform_1.apply(objectives.init_weights)
        self.cross_modal_image_transform_2 = nn.Linear(512, config['hidden_size'])
        self.cross_modal_image_transform_2.apply(objectives.init_weights)

        # convnext
        self.convnexts = getattr(convnext, 'convnext_base')(pretrained=True, in_22k=True, num_classes=21841)

        self.fc1 = nn.Linear(256, 768)
        self.fc2 = nn.Linear(512, 768)
        self.fc3 = nn.Linear(1024, 768)

        act = config['activation']
        # add Textual + Visual Encoder
        self.txt_enc = Textual_Enconder(textual_dim=768, factor=768, act=act)
        self.img_enc = Visual_Enconder(visual_dim=1024, factor=768, act=act)

        self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        freeze_layers(self.text_transformer.encoder, False)
        freeze_layers(self.text_transformer.embeddings, False)
        freeze_layers(self.text_transformer.pooler, False)
        freeze_layers(self.convnexts, False)

    def adjust_k(self):
        """
            Update loss hyper-parameter k
            linearly from intial_k to 1 according to
            the number of epochs
        """
        self.iteration += 1

        if self.max_violation:
            self.k = 1
            return 1.

        self.k = (1.-self.beta**np.float(self.iteration))
        return self.k

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_types = batch[f"text_types"]
        text_masks = batch[f"text_masks"]
        text_lengths = batch[f'lengths']

        # pooled_output: (64, 768)
        all_encoder_layers, pooled_output = self.text_transformer(text_ids, token_type_ids=text_types, attention_mask=text_masks)

        text_embeds = all_encoder_layers[-1]
        text_embeds = self.cross_modal_text_transform(text_embeds)
        text_embeds_2 = all_encoder_layers[9]
        text_embeds_2 = self.cross_modal_text_transform_2(text_embeds_2)

        # v4: convnexts
        image_embeds_global, image_embeds_all = self.convnexts(img)  # (B, 1024)
        image_embeds_2 = image_embeds_all[2]  # (B, 512, 14, 14)
        image_embeds_2 = image_embeds_2.reshape(image_embeds_2.size(0), image_embeds_2.size(1), -1).permute(0, 2, 1)  # (B, 196, 512) -> (B, 196, 768)
        image_embeds_2 = self.fc2(image_embeds_2)

        image_embeds = image_embeds_all[3]  # (B, 1024, 7, 7)
        image_embeds = image_embeds.reshape(image_embeds.size(0), image_embeds.size(1), -1).permute(0, 2, 1)  # (B, 49, 1024) -> (B, 49, 768)
        image_embeds = self.fc3(image_embeds)

        full_cap_emb_aggr = self.txt_enc(pooled_output)         # (B, 768) - > (B, 768)
        full_img_emb_aggr = self.img_enc(image_embeds_global)   # (64, 1024) - > (B, 768)

        ret = {
            "text_feats_h": text_embeds,
            "image_feats_h": image_embeds,
            "text_feats_m": text_embeds_2,
            'image_feats_m': image_embeds_2,
            "text_ids": text_ids,
            "text_masks": text_masks,
            'text_lengths': text_lengths,
            'full_img_emb_aggr': full_img_emb_aggr,
            'full_cap_emb_aggr': full_cap_emb_aggr,
        }
        return ret


    def collate_function_both(self, batch):
        cap_node_albef = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        cap_node_dot = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        list_cap_edge_index = []
        list_cap_edge_attr = []
        cap_cls_albef = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        cap_cls_dot = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        cap_cls_albef_ori = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        cap_cls_dot_ori = torch.tensor(()).to(batch[0]['img']['node_albef'].device)

        img_node_albef = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        img_node_dot = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        list_img_edge_index = []
        list_img_edge_attr = []
        img_cls_albef = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        img_cls_dot = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        img_cls_albef_ori = torch.tensor(()).to(batch[0]['img']['node_albef'].device)
        img_cls_dot_ori = torch.tensor(()).to(batch[0]['img']['node_albef'].device)

        list_id = []
        list_n_img_node = []
        list_n_img_node_albef = []
        list_n_img_node_dot = []
        list_n_cap_node = []
        list_n_cap_node_albef = []
        list_n_cap_node_dot = []
        for x in batch:
            img_cls_albef = torch.cat((img_cls_albef, x['img']['cls_albef']), dim=0)
            img_cls_dot = torch.cat((img_cls_dot, x['img']['cls_dot']), dim=0)
            img_node_albef = torch.cat((img_node_albef, x['img']['node_albef']), dim=0)
            img_node_dot = torch.cat((img_node_dot, x['img']['node_dot']), dim=0)
            list_img_edge_index.append(x['img']['edge_index'])
            list_img_edge_attr.append(x['img']['edge_attr'])
            list_n_img_node.append(x['img']['node_albef'].shape[0] + x['img']['node_dot'].shape[0])
            list_n_img_node_albef.append(x['img']['node_albef'].shape[0])
            list_n_img_node_dot.append(x['img']['node_dot'].shape[0])
            img_cls_albef_ori = torch.cat((img_cls_albef_ori, x['img']['cls_albef_ori']), dim=0)
            img_cls_dot_ori = torch.cat((img_cls_dot_ori, x['img']['cls_dot_ori']), dim=0)

            cap_cls_albef = torch.cat((cap_cls_albef, x['cap']['cls_albef']), dim=0)
            cap_cls_dot = torch.cat((cap_cls_dot, x['cap']['cls_dot']), dim=0)
            cap_node_albef = torch.cat((cap_node_albef, x['cap']['node_albef']), dim=0)
            cap_node_dot = torch.cat((cap_node_dot, x['cap']['node_dot']), dim=0)
            list_cap_edge_index.append(x['cap']['edge_index'])
            list_cap_edge_attr.append(x['cap']['edge_attr'])
            list_n_cap_node.append(x['cap']['node_albef'].shape[0] + x['cap']['node_dot'].shape[0])
            list_n_cap_node_albef.append(x['cap']['node_albef'].shape[0])
            list_n_cap_node_dot.append(x['cap']['node_dot'].shape[0])
            cap_cls_albef_ori = torch.cat((cap_cls_albef_ori, x['cap']['cls_albef_ori']), dim=0)
            cap_cls_dot_ori = torch.cat((cap_cls_dot_ori, x['cap']['cls_dot_ori']), dim=0)
            list_id.append(x['id'])

        bs = len(list_id)
        img_edge_attr = torch.cat(list_img_edge_attr).to(batch[0]['img']['node_albef'].device)
        cap_edge_attr = torch.cat(list_cap_edge_attr).to(batch[0]['img']['node_albef'].device)
        del list_img_edge_attr, list_cap_edge_attr
        img_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_img_node)).to(batch[0]['img']['node_albef'].device)
        cap_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_cap_node)).to(batch[0]['img']['node_albef'].device)
        count_img = 0
        count_cap = 0
        for idx in range(bs):
            list_img_edge_index[idx] = list_img_edge_index[idx] + count_img
            list_cap_edge_index[idx] = list_cap_edge_index[idx] + count_cap
            count_img += list_n_img_node[idx]
            count_cap += list_n_cap_node[idx]
        img_edge_index = torch.cat(list_img_edge_index, dim=1).to(batch[0]['img']['node_albef'].device)
        cap_edge_index = torch.cat(list_cap_edge_index, dim=1).to(batch[0]['img']['node_albef'].device)
        del list_img_edge_index, list_cap_edge_index
        n_img_node_albef = torch.tensor(list_n_img_node_albef).to(batch[0]['img']['node_albef'].device)
        n_img_node_dot = torch.tensor(list_n_img_node_dot).to(batch[0]['img']['node_albef'].device)
        n_cap_node_albef = torch.tensor(list_n_cap_node_albef).to(batch[0]['img']['node_albef'].device)
        n_cap_node_dot = torch.tensor(list_n_cap_node_dot).to(batch[0]['img']['node_albef'].device)
        del list_n_img_node_albef, list_n_img_node_dot, list_n_cap_node_albef, list_n_cap_node_dot
        img_dict = {'cls_albef': img_cls_albef, 'cls_dot': img_cls_dot, 'batch_index': img_batch_index,
                    'node_albef': img_node_albef, 'node_dot': img_node_dot,
                    'n_node_albef': n_img_node_albef, 'n_node_dot': n_img_node_dot,
                    'edge_index': img_edge_index, 'edge_attr': img_edge_attr,
                    'cls_albef_ori': img_cls_albef_ori, 'cls_dot_ori': img_cls_dot_ori}
        cap_dict = {'cls_albef': cap_cls_albef, 'cls_dot': cap_cls_dot, 'batch_index': cap_batch_index,
                    'node_albef': cap_node_albef, 'node_dot': cap_node_dot,
                    'n_node_albef': n_cap_node_albef, 'n_node_dot': n_cap_node_dot,
                    'edge_index': cap_edge_index, 'edge_attr': cap_edge_attr,
                    'cls_albef_ori': cap_cls_albef_ori, 'cls_dot_ori': cap_cls_dot_ori}
        list_id = torch.tensor([[int(x.split('_')[0]) for x in list_id]]).reshape(-1, 1).to(batch[0]['img']['node_albef'].device)
        return img_dict, cap_dict, list_id

    def create_index_from_2_list(self, list_1, list_2, dual_index=False, self_loop=False):
        first = np.repeat(list_1, len(list_2))
        second = np.tile(list_2, len(list_1))
        result = np.asarray([first, second])
        if dual_index:
            first = np.repeat(list_2, len(list_1))
            second = np.tile(list_1, len(list_2))
            result = np.concatenate((result, np.asarray([first, second])), axis=1)
        if self_loop:
            list_all = list_1 + list_2
            result = np.concatenate((result, np.asarray([list_all, list_all])), axis=1)
        return result

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr_my(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        self.eval_bool = True

        meter_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        pass
        '''meter_utils.set_task(self)
        output = self(batch)'''

    def validation_epoch_end(self, outs):
        if self.current_epoch!=0:
            meter_utils.epoch_eval_irtr(self)

        if self.current_epoch >= 10:
            freeze_layers(self.convnexts, True)
            freeze_layers(self.text_transformer.encoder, True)
            freeze_layers(self.text_transformer.embeddings, True)
            freeze_layers(self.text_transformer.pooler, True)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outs):
        #meter_utils.epoch_eval_irtr(self)
        meter_utils.epoch_eval_irtr(self, is_test=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
