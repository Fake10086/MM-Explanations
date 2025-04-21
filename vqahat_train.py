import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
from PIL import Image
from accelerate import Accelerator
from utils import data_utils
from utils.data_utils import *
from transformers.modeling_utils import Conv1D
from scipy.stats import spearmanr, rankdata
from transformers.activations import ACT2FN
from models.clip_vit import ImageEncoder
from transformers import AdamW, get_linear_schedule_with_warmup
from models.gpt import GPT2LMHeadModel
from transformers import GPT2Tokenizer, AutoConfig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from accelerate import DistributedDataParallelKwargs


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.cross_att.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr':2e-5},
        {'params': [p for n, p in model.cross_att.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr':2e-5},
        {'params': [p for n, p in model.Seg_net.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, "lr": 1e-4},
        {'params': [p for n, p in model.Seg_net.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, "lr": 1e-4}
    ]
    '''
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.Seg_net.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, "lr": 1e-4},
        {'params': [p for n, p in model.Seg_net.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, "lr": 1e-4}
    ]
    '''
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

def save_model(epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, save_path):
    '''
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
    '''
    save_path = save_path + 'vqahat_epoch_{}.pt'.format(str(epoch))
    torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'scheduler_state_dict': scheduler_state_dict,
                }, save_path)



def load_model(Model, path):
    '''
    model = TheModelClass(*args, **kwargs)
    optimizer = TheOptimizerClass(*args, **kwargs)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    # - or -
    model.train()
    '''

    checkpoint = torch.load(path)
    new_state = {}
    for k, v in checkpoint['model_state_dict'].items():
        if "module" in k:
            new_name = k[7:]
        else:
            new_name = k
        new_state[new_name] = v
    #model
    model = Model
    model.load_state_dict(new_state)

    #optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.cross_att.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr':2e-5},
        {'params': [p for n, p in model.cross_att.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr':2e-5},
        {'params': [p for n, p in model.gpt2_transformer.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': 1e-5},
        {'params': [p for n, p in model.gpt2_transformer.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-5},
        {'params': [p for n, p in model.Seg_net.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, "lr": 1e-4},
        {'params': [p for n, p in model.Seg_net.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, "lr": 1e-4}
    ]
    #optimizer_grouped_parameters = checkpoint['optimizer_state_dict']['param_groups']
    optimizer = AdamW(optimizer_grouped_parameters)

    #scheduler
    t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
    warmup_steps = 0  # 0.10 * t_total
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    epoch = checkpoint['epoch'] + 1

    model.eval()
    return model, optimizer, scheduler, epoch

class VQAXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len, max_exp_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.max_exp_seq_len = max_exp_seq_len
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())

        for k, v in self.data.items():
            if 'explanation' in v.keys():
                if len(v['explanation']) > 1:  # some questions have more than one explanation
                    # duplicate them for loading. -1 because one explanation is already in ids_list
                    self.ids_list += [str(k)] * (len(v['explanation']) - 1)


        self.index_tracker = {k: len(v['explanation']) - 1 for k, v in self.data.items() if 'explanation' in v.keys()}

    def __getitem__(self, i):

        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = data_utils.proc_ques(sample['question'])  # question
        answer = data_utils.proc_ans(sample['answers'])
        if 'explanation' in sample.keys():
            exp_idx = self.index_tracker[
                quention_id]  # the index of the explanation for questions with multiple explanations
            if exp_idx > 0:
                self.index_tracker[quention_id] -= 1  # decrease usage

            exp_text = sample['explanation'][exp_idx]  # explanation

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(
            ['<question>', '<answer>', '<explanation>'])
        ques_tokens = self.tokenizer.tokenize(text_a)
        ques_labels = [-100] * len(ques_tokens)  # we dont want to predict the question, set to pad to ignore in XE
        ques_segment_ids = [q_segment_id] * len(ques_tokens)

        # concat questions and answers for upper stream
        ans_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is " + answer) + [
            self.tokenizer.eos_token]
        answer_len = len(ans_tokens)
        tokens4ques_ans = ques_tokens + ans_tokens

        # concat questions and explanations for lower stream
        if 'explanation' in sample.keys():
            exp_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" because " + exp_text) + [
                self.tokenizer.eos_token]
            exp_len = len(exp_tokens)
            tokens4ques_exp = ques_tokens + exp_tokens
        else:
            exp_tokens = [self.tokenizer.bos_token]
            exp_len = len(exp_tokens)
            tokens4ques_exp = ques_tokens + exp_tokens


        # labels will be shifted in the model, so for now set them same as tokens
        labels4ques_ans = ques_labels + [-100] + ans_tokens[1:]
        labels4ques_exp = ques_labels + [-100] + exp_tokens[1:]
        ques_ans_segment_ids = ques_segment_ids + [a_segment_id] * answer_len
        ques_exp_segment_ids = ques_segment_ids + [e_segment_id] * exp_len

        # for ques_ans output
        if len(tokens4ques_ans) > self.max_seq_len:
            tokens4ques_ans = tokens4ques_ans[:self.max_seq_len]
            labels4ques_ans = labels4ques_ans[:self.max_seq_len]
            ques_ans_segment_ids = ques_ans_segment_ids[:self.max_seq_len]

        assert len(tokens4ques_ans) == len(labels4ques_ans)
        assert len(tokens4ques_ans) == len(ques_ans_segment_ids)

        ques_ans_seq_len = len(tokens4ques_ans)
        padding_len = self.max_seq_len - ques_ans_seq_len
        tokens4ques_ans += ([self.tokenizer.pad_token] * padding_len)
        labels4ques_ans += ([-100] * padding_len)
        ques_ans_segment_ids += ([a_segment_id] * padding_len)

        input_ques_ans_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_ans)
        input_ques_ans_ids = torch.tensor(input_ques_ans_ids, dtype=torch.long)

        labels4ques_ans = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels4ques_ans]
        labels4ques_ans = torch.tensor(labels4ques_ans, dtype=torch.long)

        ques_ans_segment_ids = torch.tensor(ques_ans_segment_ids, dtype=torch.long)

        # for ques_exp output
        if len(tokens4ques_exp) >= self.max_seq_len:
            tokens4ques_exp = tokens4ques_exp[:self.max_seq_len - 1]
            labels4ques_exp = labels4ques_exp[:self.max_seq_len - 1]
            ques_exp_segment_ids = ques_exp_segment_ids[:self.max_seq_len -1]

        assert (len(tokens4ques_exp)) == len(labels4ques_exp)
        assert (len(tokens4ques_exp)) == len(ques_exp_segment_ids)

        ques_exp_seq_len = len(tokens4ques_exp)
        padding_len = self.max_exp_seq_len - ques_exp_seq_len
        if padding_len > 0:
            tokens4ques_exp += ([self.tokenizer.pad_token] * (padding_len - 1))
            labels4ques_exp += ([-100] * (padding_len - 1)) # one more position for answer weight
            ques_exp_segment_ids += ([e_segment_id] * (padding_len - 1))

        labels4ques_exp = [-100] + labels4ques_exp

        input_ques_exp_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_exp)
        input_ques_exp_ids = torch.tensor(input_ques_exp_ids, dtype=torch.long)

        labels4ques_exp = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels4ques_exp]
        labels4ques_exp = torch.tensor(labels4ques_exp, dtype=torch.long)

        ques_exp_segment_ids = torch.tensor(ques_exp_segment_ids, dtype=torch.long)

        #for ques_vis output
        tokens4ques_vis = ques_tokens
        labels4ques_vis = ques_labels
        ques_vis_segment_ids = ques_segment_ids
        if len(tokens4ques_vis) >= self.max_seq_len:
            tokens4ques_vis = tokens4ques_vis[:self.max_seq_len - 1]
            labels4ques_vis = labels4ques_vis[:self.max_seq_len - 1]
            ques_vis_segment_ids = ques_segment_ids[:self.max_seq_len -1]

        assert (len(tokens4ques_vis)) == len(labels4ques_vis)
        assert (len(tokens4ques_vis)) == len(ques_vis_segment_ids)

        ques_vis_seq_len = len(tokens4ques_vis)
        padding_len = self.max_seq_len - ques_vis_seq_len
        if padding_len > 0:
            tokens4ques_vis += ([self.tokenizer.pad_token] * (padding_len))
            labels4ques_vis += ([-100] * (padding_len))
            ques_vis_segment_ids += ([e_segment_id] * (padding_len))

        #labels4ques_vis = [-100] + labels4ques_vis

        input_ques_vis_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_vis)
        input_ques_vis_ids = torch.tensor(input_ques_vis_ids, dtype=torch.long)

        labels4ques_vis = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels4ques_vis]
        labels4ques_vis = torch.tensor(labels4ques_vis, dtype=torch.long)

        ques_vis_segment_ids = torch.tensor(ques_vis_segment_ids, dtype=torch.long)

        '''
        if 'explanation' not in sample.keys():
            igonred_token = torch.full([1], 50257)
            exp_token_for_vision = torch.full([1], 50260)
            input_question_ids_for_vision = torch.cat((input_ques_exp_ids, igonred_token), dim=0)
            labels_for_vision = labels4ques_exp
            question_segment_ids_for_vision = torch.cat((ques_exp_segment_ids, exp_token_for_vision), dim=0)

        else:
            input_question_ids_for_vision =
            labels_for_vision =
            question_segment_ids_for_vision =
        '''

        folder = '/home/intership/nlxgpt/images/train2014/' if 'train' in img_name else '/home/intership/nlxgpt/images/val2014/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])

        if 'question_id' in sample.keys():
            mask_folder = '/home/intership/nlxgpt/mask/vqahat_train/' if 'train' in img_name else '/home/intership/nlxgpt/mask/vqahat_val/'
            mask_name = str(sample['question_id']) + '_1.png'
            mask_path = mask_folder + mask_name
            mask = Image.open(mask_path)
            mask_transform = transforms.Compose([transforms.Resize((mask_size, mask_size)),
                                                 transforms.ToTensor()])
            transform_mask = mask_transform(mask)


            #to_tensor
            #transform_mask = np.array(transform_mask)
            #transform_mask = torch.tensor(transform_mask, dtype=torch.float)

            #softmax, maybe try log_softmax later
            #soft_max = nn.Softmax(dim=1)
            #transform_mask = transform_mask.reshape([1, 196])


            '''
            #test
            t = transform_mask.reshape([14,14])
            s = soft_max(transform_mask).reshape([14,14])
            f = F.softmax(transform_mask, dim=1).reshape([14,14])
            l = F.log_softmax(transform_mask, dim=1).reshape([14,14])
            max = transform_mask.max()
            m = (transform_mask / max).reshape([14,14])
            
            
            transform_mask = soft_max(transform_mask)
            '''

        else:
            transform_mask = torch.full([1,14,14],255)

        return (
        img, qid, input_ques_ans_ids, input_ques_exp_ids, input_ques_vis_ids, labels4ques_ans, labels4ques_exp, labels4ques_vis, ques_ans_segment_ids,
        ques_exp_segment_ids, ques_vis_segment_ids, transform_mask)

    def __len__(self):
        return len(self.ids_list)



class VQAHATEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer):
        self.data = json.load(open(path, 'r'))
        self.transform = transform
        self.ids_list = list(self.data.keys())
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ids_list)


    def __getitem__(self, i):
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = data_utils.proc_ques(sample['question'])

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(
            ['<question>', '<answer>', '<explanation>'])
        ques_tokens = self.tokenizer.tokenize(text_a)
        ques_segment_ids = [q_segment_id] * len(ques_tokens)
        input_ques_vis_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)

        ques_segment_ids = torch.tensor(ques_segment_ids, dtype=torch.long)
        input_ques_vis_ids = torch.tensor(input_ques_vis_ids, dtype=torch.long)

        folder = '/home/intership/nlxgpt/images/train2014/' if 'train' in img_name else '/home/intership/nlxgpt/images/val2014/'  # test and val are both in val2014
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])

        image4seg = Image.open(img_path).convert('RGB')
        question4seg = [text_a]

        mask_folder = '/home/intership/nlxgpt/mask/vqahat_train/' if 'train' in img_name else '/home/intership/nlxgpt/mask/vqahat_val/'
        mask_name1 = str(sample['question_id']) + '_1.png'
        mask_path1 = mask_folder + mask_name1
        mask_name2 = str(sample['question_id']) + '_2.png'
        mask_path2 = mask_folder + mask_name2
        mask_name3 = str(sample['question_id']) + '_3.png'
        mask_path3 = mask_folder + mask_name3
        mask1 = Image.open(mask_path1)
        mask2 = Image.open(mask_path2)
        mask3 = Image.open(mask_path3)
        mask_transform = transforms.Compose([transforms.Resize((mask_size, mask_size)),
                                             transforms.ToTensor()])

        transform_mask1 = mask_transform(mask1)
        #transform_mask1 = np.array(transform_mask1)
        #transform_mask1 = torch.tensor(transform_mask1, dtype=torch.float)
        #transform_mask1 = transform_mask1.reshape([1,196])

        transform_mask2 = mask_transform(mask2)
        #transform_mask2 = np.array(transform_mask2)
        #transform_mask2 = torch.tensor(transform_mask2, dtype=torch.float)
        #transform_mask2 = transform_mask2.reshape([1,196])

        transform_mask3 = mask_transform(mask3)
        #transform_mask3 = np.array(transform_mask3)
        #transform_mask3 = torch.tensor(transform_mask3, dtype=torch.float)
        #transform_mask3 = transform_mask3.reshape([1,196])

        '''
        soft_max = nn.Softmax(dim=1)
        transform_mask1 = soft_max(transform_mask1)
        transform_mask2 = soft_max(transform_mask2)
        transform_mask3 = soft_max(transform_mask3)
        '''

        return (
            img, qid, transform_mask1, transform_mask2, transform_mask3, input_ques_vis_ids, ques_segment_ids, img_path
        )




class Attention(nn.Module):

    def __init__(self, is_cross_attention=None):
        super(Attention, self).__init__()
        self.embed_dim = 768
        self.split_size = self.embed_dim
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention == True:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_dropout = nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.resid_pdrop)
        self.num_heads = 12
        self.head_dim = 64
        self.scale_attn_weights = True


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value):

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self, hidden_states, encoder_hidden_states=None):

        if self.is_cross_attention == True:
            query = self.q_attn(encoder_hidden_states)
            key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output, attn_weights = self._attn(query, key, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, hidden_state):
        super().__init__()
        embed_dim = hidden_state
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN["gelu_new"]
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_state = 768
        inner_dim = 4 * self.hidden_state

        #self.wte = nn.Embedding(50261, self.hidden_state)
        #self.wpe = nn.Embedding(1024, self.hidden_state)

        self.ln_1 = nn.LayerNorm(self.hidden_state, eps=1e-5)
        self.self_attention = Attention(is_cross_attention=False)

        self.ln_cross_attn = nn.LayerNorm(self.hidden_state, eps=1e-5)

        self.cross_attention = Attention(is_cross_attention=True)

        self.ln_2 = nn.LayerNorm(self.hidden_state, eps=1e-5)
        self.mlp = GPT2MLP(inner_dim, self.hidden_state)


    def forward(self, hidden_states, encoder_hidden_states):

        residual = encoder_hidden_states

        encoder_hidden_states = self.ln_1(encoder_hidden_states)
        self_attn_outputs = self.self_attention(hidden_states = encoder_hidden_states)
        attn_output = self_attn_outputs

        encoder_hidden_states = residual + attn_output

        residual = encoder_hidden_states

        encoder_hidden_states = self.ln_cross_attn(encoder_hidden_states)
        cross_attn_outputs = self.cross_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states
        )

        attn_output = cross_attn_outputs
        # residual connection
        encoder_hidden_states = residual + attn_output

        residual = encoder_hidden_states

        encoder_hidden_states = self.ln_2(encoder_hidden_states)
        feed_forward_hidden_states = self.mlp(encoder_hidden_states)
        # residual connection
        encoder_hidden_states = residual + feed_forward_hidden_states

        return encoder_hidden_states

class CrossAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 768
        self.vocab_size = 50261
        self.max_position = 1024

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(self.max_position, self.embed_dim)

        self.drop = nn.Dropout(0.1)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=1e-5)

        #self.cross_attention_block = CrossAttentionBlock()
        self.cross_attention_blocks = nn.ModuleList([CrossAttentionBlock() for i in range(4)])

    def forward(self, input_ids, token_type_ids, encoder_hidden_states, answer_hidden_states):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        device = input_ids.device

        token_type_ids = token_type_ids.view(-1, input_shape[-1])

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds

        token_type_embeds = self.wte(token_type_ids)
        #hidden_states = hidden_states + token_type_embeds

        hidden_states = answer_hidden_states[6]

        hidden_states = self.drop(hidden_states)

        #attention_output = self.cross_attention_block(hidden_states, encoder_hidden_states)

        for i, block in enumerate(self.cross_attention_blocks):


            attention_output = block(hidden_states, encoder_hidden_states)
            #if i != 3:
                #hidden_states = answer_hidden_states[(i+1)*2]
            encoder_hidden_states = attention_output


        encoder_hidden_states = self.ln_f(attention_output)

        return encoder_hidden_states



class VQAHATModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.gpt2_transformer = gpt2_model.transformer
        self.cross_att = CrossAttentionModel()
        self.Seg_net = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1), nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())



    def mse_loss(self, input, target, ignored_index, reduction):
        mask = target == ignored_index
        out = (input[~mask]-target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "sum":

            mask_shape = input[~mask].reshape([-1,196]).shape[0]
            mask_prop = input.shape[0] / mask_shape
            out = out.sum() * mask_prop * 0.01
            return  out
        elif reduction == "None":
            return out

    def smooth_l1_loss(self, input, target, ignored_index, reduction, sigma=0.5, normalizer=1.0):
        mask = target == ignored_index

        mask_input = input[~mask].reshape([-1, 196])
        mask_target = target[~mask].reshape([-1, 196])

        beta = 1. / (sigma ** 2)
        diff = torch.abs(mask_input - mask_target)
        cond = diff < beta
        out = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)


        #out = (input[~mask]-target[~mask])**2

        if reduction == "mean":
            return out.mean() / normalizer
        elif reduction == "sum":

            #mask_shape = input[~mask].reshape([-1,196]).shape[0]
            #mask_prop = input.shape[0] / mask_shape
            #out = out.sum() * mask_prop * 0.01
            return  out.sum() / normalizer

        elif reduction == "None":
            return out / normalizer

    def bce_loss(self, input, target, ignored_index, reduction='sum'):
        mask = target == ignored_index
        mask_input = input[~mask].reshape(-1, 14, 14)
        mask_target = target[~mask].reshape(-1, 14, 14)
        bce_loss = nn.BCELoss(reduction=reduction)
        loss = 0.01 * bce_loss(mask_input, mask_target)
        return loss

    def kl_loss(self, input, target, ignored_index, reduction="batchmean"):
        mask = target == ignored_index
        mask_input = input[~mask].reshape(-1, 14, 14)
        mask_target = target[~mask].reshape(-1, 14, 14)
        kl_loss = nn.KLDivLoss(reduction=reduction)
        loss = kl_loss(torch.log(mask_input), mask_target)
        mask_shape = mask_target.shape[0]
        mask_prop = mask_shape / input.shape[0]
        loss = 100 * loss /mask_prop
        return loss

    def rank_corr(self, input, target, ignored_index):
        mask = target == ignored_index
        mask_input = input[~mask].reshape(-1, 196)
        mask_target = target[~mask].reshape(-1, 196)
        batch_rc_value = 0
        batch_rc_index = 0
        for i in range(mask_input.shape[0]):
            p1 = mask_input[i]
            p2 = mask_target[i]
            p1_index = rankdata(p1.detach().cpu().numpy())
            p2_index = rankdata(p2.detach().cpu().numpy())

            p1 = p1.detach().cpu().numpy()
            p2 = p2.detach().cpu().numpy()


            rc_value = spearmanr(p1,p2).correlation
            rc_index = spearmanr(p1_index,p2_index).correlation
            batch_rc_value += rc_value
            batch_rc_index += rc_index
        batch_rc_value = batch_rc_value / mask_input.shape[0]
        batch_rc_index = batch_rc_index / mask_input.shape[0]
        return batch_rc_value, batch_rc_index


    def forward(self, img_embeddings, input_ques_vis_ids, ques_vis_segment_ids, target):

        loss = 0

        hidden_state = self.gpt2_transformer(input_ids=input_ques_vis_ids.to(device), token_type_ids=ques_vis_segment_ids.to(device), encoder_hidden_states=img_embeddings.to(device))
        hidden_state = hidden_state['hidden_states']

        img_embeddings_with_cross_att = self.cross_att(input_ques_vis_ids, ques_vis_segment_ids, img_embeddings, hidden_state)
        #img_embeddings_with_cross_att = self.cross_att_block(hidden_states, img_embeddings)

        visual_input = torch.permute(img_embeddings_with_cross_att,(0,2,1)).contiguous().reshape([img_embeddings_with_cross_att.shape[0], img_embeddings_with_cross_att.shape[2],14, 14])
        visual_output = self.Seg_net(visual_input)
        #visual_output = visual_output.view(-1, 1, 196)
        #softmax = nn.Softmax(dim=2)
        #visual_output = softmax(visual_output)
        #mse_loss = self.smooth_l1_loss(visual_output, target, ignored_index=255, reduction='sum')
        bce_loss = self.bce_loss(visual_output, target, ignored_index=255, reduction='sum')
        rc_value, rc_index = self.rank_corr(visual_output, target, ignored_index=255)
        #kl_loss = self.kl_loss(visual_output, target, ignored_index=255)
        '''
        if torch.isnan(mse_loss) == False:
            #mse_loss = 0.01 * mse_loss
            loss += mse_loss
        '''
        loss = bce_loss
        return loss, rc_value, rc_index

def visulize_attention_ratio(img_path, attention_mask, i, ratio=0.5, cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    print("load image from: ", img_path)
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention mask
    mask = cv2.resize(attention_mask.reshape([14,14]), (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')

    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    plt.savefig('visualize/{}_{}.jpg'.format(img_path.split('/')[6].split('.')[0], i))

def evalute_mask(model, loader):
    model.eval()

    all_rank_correlation = 0
    all_rank_correlation_value = 0
    for i, batch in enumerate(loader):
        batch1 = tuple(batch[i].to(device) for i in range(7))
        img, qid, transform_mask1, transform_mask2, transform_mask3, input_ques_vis_ids, ques_segment_ids= batch1
        img_embeddings = image_encoder(img)
        img_path = batch[7][0]
        hidden_state = model.gpt2_transformer(input_ids=input_ques_vis_ids.to(device), token_type_ids=ques_segment_ids.to(device), encoder_hidden_states=img_embeddings.to(device))
        hidden_state = hidden_state['hidden_states']

        img_embeddings_with_cross_att = model.cross_att(input_ques_vis_ids, ques_segment_ids, img_embeddings, hidden_state)

        visual_input = torch.permute(img_embeddings_with_cross_att, (0, 2, 1)).contiguous().reshape(
            [img_embeddings_with_cross_att.shape[0], img_embeddings_with_cross_att.shape[2], 14, 14])
        visual_output = model.Seg_net(visual_input)

        '''
        visual_output = visual_output.reshape([196])
        transform_mask1 = transform_mask1.reshape([196])
        transform_mask2 = transform_mask2.reshape([196])
        transform_mask3 = transform_mask3.reshape([196])
        softmax = nn.Softmax(dim=0)
        visual_output = softmax(visual_output)
        '''
        visulize_attention_ratio(img_path, visual_output.detach().cpu().numpy(), 0)
        visulize_attention_ratio(img_path, transform_mask1.detach().cpu().numpy(), 1)
        visulize_attention_ratio(img_path, transform_mask2.detach().cpu().numpy(), 2)
        visulize_attention_ratio(img_path, transform_mask3.detach().cpu().numpy(), 3)
        visual_output_value = visual_output.reshape([196]).detach().cpu().numpy()
        transform_mask1_value = transform_mask1.reshape([196]).detach().cpu().numpy()
        transform_mask2_value = transform_mask2.reshape([196]).detach().cpu().numpy()
        transform_mask3_value = transform_mask3.reshape([196]).detach().cpu().numpy()


        #visual_output = rankdata(visual_output.detach().cpu().numpy())
        #transform_mask1 = rankdata(transform_mask1.detach().cpu().numpy())
        #transform_mask2 = rankdata(transform_mask2.detach().cpu().numpy())
        #transform_mask3 = rankdata(transform_mask3.detach().cpu().numpy())


        '''
        visual_output_value = torch.sort(visual_output.reshape([196]), descending=False, dim=-1).values
        transform_mask1_value = torch.sort(transform_mask1.reshape([196]), descending=False, dim=-1).values
        transform_mask2_value = torch.sort(transform_mask2.reshape([196]), descending=False, dim=-1).values
        transform_mask3_value = torch.sort(transform_mask3.reshape([196]), descending=False, dim=-1).values
        '''






        with torch.no_grad():
            rank_c1_value = spearmanr(visual_output_value, transform_mask1_value).correlation
            rank_c2_value = spearmanr(visual_output_value, transform_mask2_value).correlation
            rank_c3_value = spearmanr(visual_output_value, transform_mask3_value).correlation

            value_count = 3
            if torch.isnan(torch.tensor(rank_c1_value)):
                rank_c1_value = 0
                value_count = value_count - 1
            if torch.isnan(torch.tensor(rank_c2_value)):
                rank_c2_value = 0
                value_count = value_count - 1
            if torch.isnan(torch.tensor(rank_c3_value)):
                rank_c3_value = 0
                value_count = value_count -1
            if value_count != 0:
                mean_rank_value = (rank_c1_value + rank_c2_value + rank_c3_value) / value_count
                all_rank_correlation_value += mean_rank_value

        print("\rMean_Rank_Correlation Evaluation: Finished {}/{}. Mean_rank: {:.3f}".format(i, len(loader), mean_rank_value), end='          ')

    all_rank_correlation = all_rank_correlation / len(loader)
    all_rank_correlation_value = all_rank_correlation_value / len(loader)
    return  all_rank_correlation, all_rank_correlation_value

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
#accelerator = Accelerator()
device = accelerator.device

img_size = 224
mask_size = 14

ckpt_path = 'ckpts/'
load_path = 'ckpts/best_so_far.pt'
nle_data_train_path = 'train_HX_data.json'
vqahat_test_data_path = 'val_data_with_answer.json'

max_seq_len = 40
max_exp_seq_len = 40

batch_size = 32  # per GPU
num_train_epochs = 30
weight_decay = 0.005
learning_rate = 1e-4
gradient_accumulation_steps = 1
start_epoch = 0
temperature = 1
load_from_epoch = False

image_encoder = ImageEncoder(device).to(device)
change_requires_grad(image_encoder, False)
config = AutoConfig.from_pretrained('pretrained_model/pretrain_model_up')
gpt2_model = GPT2LMHeadModel.from_pretrained('pretrained_model/pretrain_model', config=config)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('pretrained_model/pretrain_tokenizer')

writer = SummaryWriter()

if load_from_epoch == False:
    model = VQAHATModel()
    model.cross_att.wte.load_state_dict(gpt2_model.transformer.wte.state_dict())
    model.cross_att.wpe.load_state_dict(gpt2_model.transformer.wpe.state_dict())

    model = model.to(device)
    optimizer = get_optimizer(model, learning_rate)



img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = VQAXTrainDataset(path=nle_data_train_path,
                                 transform=img_transform,
                                 tokenizer=gpt2_tokenizer,
                                 max_seq_len=max_seq_len,
                                 max_exp_seq_len=max_exp_seq_len
                                 )

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True)

vqahat_test_dataset = VQAHATEvalDataset(path=vqahat_test_data_path,
                                        transform=img_transform,
                                        tokenizer=gpt2_tokenizer)

vqahat_test_dataloader = torch.utils.data.DataLoader(vqahat_test_dataset,
                                                    batch_size=1,
                                                     shuffle=False,
                                                     pin_memory=True)


if load_from_epoch == True:
    model, optimizer, scheduler, start_epoch = load_model(VQAHATModel(), load_path)
    model.cross_att.wte.load_state_dict(gpt2_model.transformer.wte.state_dict())
    model.cross_att.wpe.load_state_dict(gpt2_model.transformer.wpe.state_dict())
    model = model.to(device)



model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
print("Model Setup Ready...")


if load_from_epoch == False:
    t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
    #warmup_steps = 0.1 * t_total  # 0.10 * t_total
    warmup_steps = 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

visual_result_old = 0


for epoch in range(start_epoch, num_train_epochs):
    '''
    model.train()
    accum_loss = 0
    accum_rc_value = 0
    accum_rc_index = 0
    for step, batch in enumerate(train_loader):
        #if step == 20:
        #    break
        batch = tuple(input_tensor.to(device) for input_tensor in batch)

        img, _, input_ques_ans_ids, input_ques_exp_ids, input_ques_vis_ids,labels4ques_ans, \
            labels4ques_exp, labels4ques_vis, ques_ans_segment_ids, ques_exp_segment_ids, ques_vis_segment_ids, mask  = batch

        img_embeddings = image_encoder(img)

        loss, batch_rc_value, batch_rc_index = model(
                                                img_embeddings=img_embeddings,
                                                input_ques_vis_ids=input_ques_vis_ids,
                                                ques_vis_segment_ids=ques_vis_segment_ids,
                                                target=mask
                                                )

        accelerator.backward(loss)
        accum_loss += loss.item()
        accum_rc_value += batch_rc_value.item()
        accum_rc_index += batch_rc_index.item()
        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accelerator.print(
                "\rEpoch {} / {}, Iter {} / {} Loss: {:.3f} RC_Value: {:.3f} RC_Index: {:.3f}".format(
                    epoch, num_train_epochs, step, len(train_loader),accum_loss,accum_rc_value,accum_rc_index), end='          ')

            writer.add_scalar("Loss: ", accum_loss, epoch * len(train_loader) + step)
            writer.add_scalar("LR: ", optimizer.state_dict()['param_groups'][0]['lr'], epoch * len(train_loader) + step)

            writer.add_scalar("RC: ", accum_rc_value, epoch * len(train_loader) + step)
            accum_loss = 0
            accum_rc_value = 0
            accum_rc_index = 0
    '''
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    #save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)

    if accelerator.is_main_process:
        visual_result_new, visual_result_value = evalute_mask(unwrapped_model, vqahat_test_dataloader)
        print('\n')
        print("\repoch: {} Mean_Rank_Correlation: {:.3f} Mean_Rank_Correlation_value: {:.3f}\n".format(epoch, visual_result_new, visual_result_value))
        if visual_result_new > visual_result_old:
            save_model(epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), ckpt_path)
            visual_result_old = visual_result_new