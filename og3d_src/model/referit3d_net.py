import math
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformers import BertConfig, BertModel

from .obj_encoder import GTObjEncoder, PcdObjEncoder, ObjColorEncoder
from .txt_encoder import GloveGRUEncoder
from .mmt_module import MMT
from .cmt_module import CMT

import numpy as np

def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(
                nn.Linear(input_size, hidden_size//2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size//2, eps=1e-12),
                nn.Dropout(dropout),
                nn.Linear(hidden_size//2, output_size)
            )

def freeze_bn(m):
    '''Freeze BatchNorm Layers'''
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()

class ReferIt3DNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        config.obj_encoder.num_obj_classes = config.num_obj_classes
        if self.config.model_type == 'gtlabel':
            self.obj_encoder = GTObjEncoder(config.obj_encoder, config.hidden_size)
        elif self.config.model_type == 'gtpcd':
            self.obj_encoder = PcdObjEncoder(config.obj_encoder)
        if self.config.obj_encoder.freeze:
            freeze_bn(self.obj_encoder)
            for p in self.obj_encoder.parameters():
                p.requires_grad = False
        if self.config.obj_encoder.freeze_bn:
            freeze_bn(self.obj_encoder)

        if self.config.obj_encoder.use_color_enc:
            self.obj_color_encoder = ObjColorEncoder(config.hidden_size, config.obj_encoder.dropout)

        if self.config.txt_encoder.type == 'gru':
            # glove embedding
            self.txt_encoder = GloveGRUEncoder(config.hidden_size, config.txt_encoder.num_layers)
        else:
            txt_bert_config = BertConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.txt_encoder.num_layers,
                num_attention_heads=12, type_vocab_size=2
            )
            self.txt_encoder = BertModel.from_pretrained(
                'bert-base-uncased', config=txt_bert_config
            )
        if self.config.txt_encoder.freeze:
            for p in self.txt_encoder.parameters():
                p.requires_grad = False
    
        mm_config = EasyDict(config.mm_encoder)
        mm_config.hidden_size = config.hidden_size
        mm_config.num_attention_heads = 12
        mm_config.dim_loc = config.obj_encoder.dim_loc
        if self.config.mm_encoder.type == 'cmt':
            self.mm_encoder = CMT(mm_config)
        elif self.config.mm_encoder.type == 'mmt':
            self.mm_encoder = MMT(mm_config)

        self.og3d_head = get_mlp_head(
            config.hidden_size, config.hidden_size, 
            1, dropout=config.dropout
        )

        if self.config.losses.obj3d_clf > 0:
            self.obj3d_clf_head = get_mlp_head(
                config.hidden_size, config.hidden_size, 
                config.num_obj_classes, dropout=config.dropout
            )
        if self.config.losses.obj3d_clf_pre > 0:
            self.obj3d_clf_pre_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )
            if self.config.obj_encoder.freeze:
                for p in self.obj3d_clf_pre_head.parameters():
                    p.requires_grad = False
        if self.config.losses.obj3d_reg > 0:
            self.obj3d_reg_head = get_mlp_head(
                config.hidden_size, config.hidden_size, 
                6, dropout=config.dropout
            )
        if self.config.losses.txt_clf > 0:
            self.txt_clf_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )

    def prepare_batch(self, batch):
        outs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                outs[key] = value.to(self.device)
            else:
                outs[key] = value
        return outs
        
    def forward(
        self, batch: dict, compute_loss=False, is_test=False,
        output_attentions=False, output_hidden_states=False, change=False, cfg=None, 
    ) -> dict:
        batch = self.prepare_batch(batch)

        if self.config.obj_encoder.freeze or self.config.obj_encoder.freeze_bn:
            freeze_bn(self.obj_encoder)
        obj_embeds = self.obj_encoder(batch['obj_fts'])
        # # import pdb
        # # pdb.set_trace()
        if change and 'new_tgt_feats' in batch.keys():
            idxs_tmp = [[i for i in range(batch['obj_fts'].shape[0])],batch['tgt_obj_idxs'].cpu().numpy().tolist()]
            # idxs_tmp = [[i for i in range(batch['obj_fts'].shape[0])],batch['anchor'].cpu().numpy().tolist()]
            change_obj = (batch['change']==1)
            change_obj = change_obj.cpu()
            idxs_tmp = (np.array(idxs_tmp)[:,change_obj]).tolist()
            obj_embeds[idxs_tmp] = batch['new_tgt_feats'][change_obj]
        if self.config.obj_encoder.freeze:
            obj_embeds = obj_embeds.detach()
        if self.config.obj_encoder.use_color_enc:
            obj_embeds = obj_embeds + self.obj_color_encoder(batch['obj_colors'])

        txt_embeds = self.txt_encoder(
            batch['txt_ids'], batch['txt_masks'],
        ).last_hidden_state
        if 'txt_ids_mask' in batch.keys():
            txt_embeds_mask = self.txt_encoder(
                batch['txt_ids_mask'], batch['txt_masks_mask'],
            ).last_hidden_state
        else:
            txt_embeds_mask = txt_embeds
        if self.config.txt_encoder.freeze:
            txt_embeds = txt_embeds.detach()
            txt_embeds_mask = txt_embeds_mask.detach()

        out_embeds = self.mm_encoder(
            txt_embeds, batch['txt_masks'], 
            obj_embeds, batch['obj_locs'], batch['obj_masks'],
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
        )

        if 'txt_masks_mask' in batch.keys():
            out_embeds_mask = self.mm_encoder(
                txt_embeds_mask, batch['txt_masks_mask'], 
                obj_embeds, batch['obj_locs'], batch['obj_masks'],
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states,
            )
        else:
            out_embeds_mask = out_embeds
        if cfg.use_mask_sent:
            out_embeds = out_embeds_mask
        
        og3d_logits = self.og3d_head(out_embeds['obj_embeds']).squeeze(2)
        og3d_logits.masked_fill_(batch['obj_masks'].logical_not(), -float('inf'))

        og3d_logits_mask = self.og3d_head(out_embeds_mask['obj_embeds']).squeeze(2)
        og3d_logits_mask.masked_fill_(batch['obj_masks'].logical_not(), -float('inf'))

        result = {
            # 'og3d_logits': 2 * og3d_logits + og3d_logits_mask,
            'og3d_logits': og3d_logits, 
            'og3d_logits_ori': og3d_logits,
            'og3d_logits_mask': og3d_logits_mask,
        }
        
        if output_attentions:
            result['all_cross_attns'] = out_embeds['all_cross_attns']
            result['all_self_attns'] = out_embeds['all_self_attns']
        if output_hidden_states:
            result['all_hidden_states'] = out_embeds['all_hidden_states']
        
        if self.config.losses.obj3d_clf > 0:
            result['obj3d_clf_logits'] = self.obj3d_clf_head(out_embeds['obj_embeds'])
        if self.config.losses.obj3d_reg > 0:
            result['obj3d_loc_preds'] = self.obj3d_reg_head(out_embeds['obj_embeds'])
        if self.config.losses.obj3d_clf_pre > 0:
            result['obj3d_clf_pre_logits'] = self.obj3d_clf_pre_head(obj_embeds)
        if self.config.losses.txt_clf > 0:
            result['txt_clf_logits'] = self.txt_clf_head(txt_embeds[:, 0])
        
        if compute_loss:
            losses = self.compute_loss(result, batch, cfg)
            return result, losses
        
        return result

    def compute_loss(self, result, batch, cfg):
        losses = {}
        total_loss = 0

        positive = (batch['negative'] == 0)
        og3d_logits = result['og3d_logits']
        og3d_logits_positive = og3d_logits[positive]
        tgt_obj_idxs_positive = batch['tgt_obj_idxs'][positive]

        # og3d_loss = F.cross_entropy(result['og3d_logits'], batch['tgt_obj_idxs'])
        og3d_loss_positive = F.cross_entropy(og3d_logits_positive, tgt_obj_idxs_positive)
        losses['og3d'] = og3d_loss_positive
        total_loss += og3d_loss_positive

        negative = (batch['negative'] == 1)
        og3d_logits_negative = og3d_logits[negative]
        og3d_logits_negative = F.softmax(og3d_logits_negative, dim=1)
        tgt_obj_idxs_negative = batch['tgt_obj_idxs'][negative].detach().cpu().numpy().tolist()
        negative_len = og3d_logits_negative.shape[0]
        if negative_len:
            idx_ls_tmp = [i for i in range(negative_len)]
            og3d_logits_negative = og3d_logits_negative[idx_ls_tmp,tgt_obj_idxs_negative]
            gt_negative = torch.zeros_like(og3d_logits_negative).cuda()
            og3d_loss_negative = F.mse_loss(og3d_logits_negative, gt_negative)
            losses['og3d_negative'] = og3d_loss_negative * self.config.losses.negative_obj3d
            # total_loss += og3d_loss_negative

        if self.config.losses.obj3d_clf > 0:
            obj3d_clf_loss = F.cross_entropy(
                result['obj3d_clf_logits'][positive].permute(0, 2, 1), 
                batch['obj_classes'][positive], reduction='none'
            )
            obj3d_clf_loss = (obj3d_clf_loss * batch['obj_masks'][positive]).sum() / batch['obj_masks'][positive].sum()
            losses['obj3d_clf'] = obj3d_clf_loss * self.config.losses.obj3d_clf
            total_loss += losses['obj3d_clf']

        if self.config.losses.obj3d_clf_pre > 0:
            obj3d_clf_pre_loss = F.cross_entropy(
                result['obj3d_clf_pre_logits'].permute(0, 2, 1), 
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_pre_loss = (obj3d_clf_pre_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum()
            losses['obj3d_clf_pre'] = obj3d_clf_pre_loss * self.config.losses.obj3d_clf_pre
            total_loss += losses['obj3d_clf_pre']

        if self.config.losses.obj3d_reg > 0:
            # import pdb
            # pdb.set_trace()
            og3d_preds = torch.argmax(result['og3d_logits'], dim=1)
            idxs_tmp = [[i for i in range(batch['obj_fts'].shape[0])], og3d_preds.cpu().numpy().tolist()]
            obj3d_reg_loss = F.mse_loss(
                result['obj3d_loc_preds'][idxs_tmp], batch['obj_locs'][idxs_tmp],  reduction='none'
            )
            # obj3d_reg_loss = (obj3d_reg_loss * batch['obj_masks'].unsqueeze(2)).sum() / batch['obj_masks'].sum()
            obj3d_reg_loss = obj3d_reg_loss.sum() / 64
            losses['obj3d_reg'] = obj3d_reg_loss * self.config.losses.obj3d_reg
            total_loss += losses['obj3d_reg']

        if self.config.losses.txt_clf > 0:
            txt_clf_loss = F.cross_entropy(
                result['txt_clf_logits'][positive], batch['tgt_obj_classes'][positive],  reduction='mean'
            )
            losses['txt_clf'] = txt_clf_loss * self.config.losses.txt_clf
            total_loss += losses['txt_clf']

        losses['total'] = total_loss
        return losses
