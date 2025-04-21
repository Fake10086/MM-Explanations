import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from PIL import Image
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils import data_utils
from utils.data_utils import *
from utils.eval_utils import top_filtering
from transformers.modeling_utils import Conv1D
from accelerate import DistributedDataParallelKwargs
from scipy.stats import spearmanr
from transformers.activations import ACT2FN
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)  # load tokenizer
    # model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    model = NLXmodel()
    # model = GPT2LMHeadModel.from_pretrained('/home/lizhengyi/2022/task_basecalling/ex_try/bonito_copy/distillgpt2', config = config)
    # model = opt['model']
    model.upper_model.resize_token_embeddings(len(tokenizer))
    model.lower_model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    opt = torch.load(ckpt_path + filename, map_location='cpu')
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch


def load_pretrained():
    model_path = 'pretrained_model/pretrain_model'
    tokenizer_path = 'pretrained_model/pretrain_tokenizer'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)  # load model with config
    return tokenizer, model


def filter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predictions):
    all_file = json.load(open(nle_data_test_path, 'r'))

    gt_answers = {}
    gt_answers_for_exp = {}
    for key, value in all_file.items():
        gt_answers_for_exp[int(key)] = data_utils.proc_ans(value['answers'])
        gt_answers[int(key)] = data_utils.proc_ans2(value['answers'])

    pred_answers = {}
    for item in full_predictions:
        # pred_answers[item['image_id']] = item['caption'].split('the answer is')[1].split("because")[0].strip()
        pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()

    correct_keys = []
    for key, value in pred_answers.items():
        gt_answer = gt_answers[key]
        # to measure accuracy for VQA, please change "==" to "in" (if value in gt_answer:)
        # you need to also change the proc_ans funtion in utils/data_uitls.py to return: list(ans_prob_dict.keys())
        if value in gt_answer:
            correct_keys.append(key)
    acc = len(correct_keys) / len(pred_answers)
    print('\n')
    print("\rAccuracy: {:.3f}\n".format(acc))

    correct_keys_for_exp = []
    for key, value in pred_answers.items():
        gt_answer = gt_answers_for_exp[key]
        # to measure accuracy for VQA, please change "==" to "in" (if value in gt_answer:)
        # you need to also change the proc_ans funtion in utils/data_uitls.py to return: list(ans_prob_dict.keys())
        if value == gt_answer:
            correct_keys_for_exp.append(key)

    exp_preds = [item for item in exp_predictions if item['image_id'] in correct_keys_for_exp]

    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)

    coco = COCO(annFileExp)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    with open(save_scores_pathExp, 'w') as w:
        json.dump(cocoEval.eval, w)


def save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)  # save tokenizer

    unwrapped_model.save_pretrained(ckpt_path, epoch)

    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           **kwargs}

    accelerator.save(opt, ckpt_path + filename)


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
        tokens4loss_ea_answer = self.tokenizer.tokenize(" the answer is " + answer) + [self.tokenizer.eos_token]

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
            tokens4loss_ea = tokens4ques_exp + tokens4loss_ea_answer # tokens4loss_ea = q + e + a

        # labels will be shifted in the model, so for now set them same as tokens
        labels4ques_ans = ques_labels + [-100] + ans_tokens[1:]
        labels4ques_exp = ques_labels + [-100] + exp_tokens[1:]
        #labels4loss_ea_answer = tokens4loss_ea_answer
        #labels4loss_ea = labels4ques_exp + labels4loss_ea_answer

        ques_ans_segment_ids = ques_segment_ids + [a_segment_id] * answer_len
        ques_exp_segment_ids = ques_segment_ids + [e_segment_id] * exp_len
        loss_ea_segment_ids = ques_exp_segment_ids

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
            ques_exp_segment_ids = ques_exp_segment_ids[:self.max_seq_len - 1]

        assert (len(tokens4ques_exp)) == len(labels4ques_exp)
        assert (len(tokens4ques_exp)) == len(ques_exp_segment_ids)

        ques_exp_seq_len = len(tokens4ques_exp)
        padding_len = self.max_exp_seq_len - ques_exp_seq_len
        if padding_len > 0:
            tokens4ques_exp += ([self.tokenizer.pad_token] * (padding_len - 1))
            labels4ques_exp += ([-100] * (padding_len - 1))  # one more position for answer weight
            ques_exp_segment_ids += ([e_segment_id] * (padding_len - 1))

        labels4ques_exp = [-100] + labels4ques_exp

        input_ques_exp_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_exp)
        input_ques_exp_ids = torch.tensor(input_ques_exp_ids, dtype=torch.long)

        labels4ques_exp = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels4ques_exp]
        if len(exp_tokens[1:]) == 0:
            labels4ques_exp = [255] * 40
        labels4ques_exp = torch.tensor(labels4ques_exp, dtype=torch.long)

        ques_exp_segment_ids = torch.tensor(ques_exp_segment_ids, dtype=torch.long)

        # for ques_vis output
        tokens4ques_vis = ques_tokens
        labels4ques_vis = ques_labels
        ques_vis_segment_ids = ques_segment_ids
        if len(tokens4ques_vis) >= self.max_seq_len:
            tokens4ques_vis = tokens4ques_vis[:self.max_seq_len - 1]
            labels4ques_vis = labels4ques_vis[:self.max_seq_len - 1]
            ques_vis_segment_ids = ques_segment_ids[:self.max_seq_len - 1]

        assert (len(tokens4ques_vis)) == len(labels4ques_vis)
        assert (len(tokens4ques_vis)) == len(ques_vis_segment_ids)

        ques_vis_seq_len = len(tokens4ques_vis)
        padding_len = self.max_seq_len - ques_vis_seq_len
        if padding_len > 0:
            tokens4ques_vis += ([self.tokenizer.pad_token] * (padding_len))
            labels4ques_vis += ([-100] * (padding_len))
            ques_vis_segment_ids += ([e_segment_id] * (padding_len))

        # labels4ques_vis = [-100] + labels4ques_vis

        input_ques_vis_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_vis)
        input_ques_vis_ids = torch.tensor(input_ques_vis_ids, dtype=torch.long)

        labels4ques_vis = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels4ques_vis]
        # labels4ques_vis = [255] * 40
        labels4ques_vis = torch.tensor(labels4ques_vis, dtype=torch.long)

        ques_vis_segment_ids = torch.tensor(ques_vis_segment_ids, dtype=torch.long)

        # for Loss_ea output




        folder = '/home/intership/nlxgpt/images/train2014/' if 'train' in img_name else '/home/intership/nlxgpt/images/val2014/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])

        return (
            img, qid, input_ques_ans_ids, input_ques_exp_ids, labels4ques_ans, labels4ques_exp, ques_ans_segment_ids,
            ques_exp_segment_ids)

    def __len__(self):
        return len(self.ids_list)


class VQAXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> because <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())

    def __getitem__(self, i):
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = data_utils.proc_ques(sample['question'])  # question

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(
            ['<question>', '<answer>', '<explanation>'])
        ques_tokens = self.tokenizer.tokenize(text_a)
        ques_segment_ids = [q_segment_id] * len(ques_tokens)

        ans_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is")
        answer_len = len(ans_tokens)
        tokens4ques_ans = ques_tokens + ans_tokens

        ques_ans_segment_ids = ques_segment_ids + [a_segment_id] * answer_len

        input_ques_ans_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_ans)
        input_ques_ans_ids = torch.tensor(input_ques_ans_ids, dtype=torch.long)
        ques_ans_segment_ids = torch.tensor(ques_ans_segment_ids, dtype=torch.long)

        exp_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" because")
        exp_len = len(exp_tokens)
        tokens4ques_exp = ques_tokens + exp_tokens

        ques_exp_segment_ids = ques_segment_ids + [e_segment_id] * exp_len

        input_ques_exp_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_exp)
        input_ques_exp_ids = torch.tensor(input_ques_exp_ids, dtype=torch.long)
        ques_exp_segment_ids = torch.tensor(ques_exp_segment_ids, dtype=torch.long)

        folder = '/home/intership/nlxgpt/images/train2014/' if 'train' in img_name else '/home/intership/nlxgpt/images/val2014/'  # test and val are both in val2014
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])

        return (img, qid, input_ques_ans_ids, input_ques_exp_ids, ques_ans_segment_ids, ques_exp_segment_ids)

    def __len__(self):
        return len(self.ids_list)


class NLXmodel(nn.Module):

    def __init__(self):
        super(NLXmodel, self).__init__()
        config_up = AutoConfig.from_pretrained('pretrained_model/pretrain_model_up')
        config_down = AutoConfig.from_pretrained('pretrained_model/pretrain_model_down')
        # Add configs
        setattr(config_up, 'img_size', None)
        setattr(config_up, 'max_seq_len', None)
        config_up.img_size = img_size
        config_up.max_seq_len = max_seq_len
        config_up.add_cross_attention = True
        setattr(config_down, 'img_size', None)
        setattr(config_down, 'max_seq_len', None)
        config_down.img_size = img_size
        config_down.max_seq_len = max_seq_len
        config_down.add_cross_attention = True
        # ckpt_path + model_name
        self.upper_model = GPT2LMHeadModel.from_pretrained('pretrained_model/pretrain_model_up', config=config_up)
        self.lower_model = GPT2LMHeadModel.from_pretrained('pretrained_model/pretrain_model_down', config=config_down)

    def save_pretrained(self, ckpt_path, epoch):
        upper_model_name = 'nle_upper_model_{}'.format(str(epoch))
        lower_model_name = 'nle_lower_model_{}'.format(str(epoch))

        self.upper_model.save_pretrained(ckpt_path + upper_model_name, save_function=accelerator.save)
        self.lower_model.save_pretrained(ckpt_path + lower_model_name, save_function=accelerator.save)

    def generate_beam(self, tokenizer, prompt_ids, prompt_seg_ids, img_embeddings, up_hidden_states, beam_size, max_len = 10):

        beam_text_1 = []
        beam_text_2 = []
        SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
        special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        new_exp_segment = special_tokens_ids[-1]
        new_exp_segment = torch.LongTensor([new_exp_segment]).to(device)
        scores = None
        tokens = None

        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        scores_list = None

        text_embedding = self.lower_model.transformer.wte(prompt_ids)
        input_embedding = text_embedding

        for step in range(max_len + 1):
            if step == max_len:
                break

            outputs = self.lower_model(
                up_hidden_states=up_hidden_states,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=prompt_seg_ids,
                position_ids=None,
                inputs_embeds=input_embedding,
                encoder_hidden_states=img_embeddings,
                encoder_attention_mask=None,
                labels=None,
                use_cache=False,
                return_dict=True,
                test_or_train='test',
                step=step
            )

            hidden_states = outputs.hidden_states[0]
            beam_logits = outputs.logits[0]
            beam_logits = beam_logits[:, -1, :] / temperature
            beam_logits = beam_logits.softmax(-1) + 1e-16
            beam_logits = beam_logits.log()
            # beam_logits_exp = torch.exp(beam_logits)
            # partition = beam_logits_exp.sum(dim=-1, keepdim=True)
            # beam_logits = beam_logits_exp / partition
            # beam_logits = torch.clamp(beam_logits, min=-50)
            if scores is None:
                scores, next_token = beam_logits.topk(beam_size, -1)
                # scores = scores.log()
                input_embedding = input_embedding.expand(beam_size, *input_embedding.shape[1:])
                prompt_seg_ids = prompt_seg_ids.expand(beam_size, *prompt_seg_ids.shape[1:])
                # img_embeddings = img_embeddings.expand(beam_size, *img_embeddings.shape[1:])
                new_exp_segment = new_exp_segment.expand(beam_size, *new_exp_segment.shape[1:])
                next_token, scores = next_token.permute(1, 0), scores.squeeze(0)
                scores_list = scores.unsqueeze(-1)
                if tokens is None:
                    tokens = next_token

                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_token), dim=1)

            else:
                beam_logits[is_stopped] = -float(np.inf)
                beam_logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + beam_logits
                # print('\nscores[:None]:{}\n'.format(scores[:,None]))
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_token = scores_sum_average.view(-1).topk(beam_size, -1)
                next_token_source = next_token // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_token_source]
                next_token = next_token % scores_sum.shape[1]
                # if torch.isnan(beam_logits[next_token_source, next_token]).any():
                #    _ = 1
                # else:
                scores_list = torch.cat(
                    [scores_list[next_token_source], beam_logits[next_token_source, next_token].unsqueeze(-1)], dim=-1)
                next_token = next_token.unsqueeze(1)
                tokens = tokens[next_token_source]
                tokens = torch.cat((tokens, next_token), dim=1)
                input_embedding = input_embedding[next_token_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_token_source]

            next_token_embedding = self.lower_model.transformer.wte(next_token.squeeze()).view(
                input_embedding.shape[0], 1, -1)
            input_embedding = torch.cat((input_embedding, next_token_embedding), dim=1)
            prompt_seg_ids = torch.cat((prompt_seg_ids, new_exp_segment.unsqueeze(1)), dim=1)

            # is_stopped = is_stopped + next_token.eq(special_tokens_ids[0]).squeeze()
            for i in range(2):
                is_stopped[i] = is_stopped[i] + torch.tensor(next_token[i].item() in special_tokens_ids)
            if is_stopped.all():
                break

        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [
            tokenizer.decode(output[: int(length) - 1]).strip()
            for output, length in zip(output_list, seq_lengths)
        ]
        # token = []
        # token.append(torch.tensor(tokenizer(output_texts[0])['input_ids'], dtype=torch.int64).to(device))
        scores = -torch.mean(scores_list[0, :tokens.shape[1]]).unsqueeze(0)
        for i in range(1, beam_size):
            # token.append(torch.tensor(tokenizer(output_texts[i])['input_ids'], dtype=torch.int64).to(device))
            scores = torch.cat((scores, -torch.mean(scores_list[i, :tokens.shape[1]]).unsqueeze(0)), dim=0)

        return output_texts, scores

    def forward(self, img_embeddings, input_ques_ans_ids, input_ques_exp_ids, labels4ques_ans, labels4ques_exp,
                ques_ans_segment_ids, ques_exp_segment_ids, test_or_train, reinforcement_flag=False):
        #start = time.time()
        upper_outputs = self.upper_model(input_ids=input_ques_ans_ids,
                                         past_key_values=None,
                                         attention_mask=None,
                                         token_type_ids=ques_ans_segment_ids,
                                         position_ids=None,
                                         encoder_hidden_states=img_embeddings,
                                         encoder_attention_mask=None,
                                         labels=labels4ques_ans,
                                         use_cache=False,
                                         return_dict=True,
                                         test_or_train='train')

        output_embedding = upper_outputs.output_embedding.weight

        prediction_list = (labels4ques_ans.cpu().detach().numpy().tolist())
        answer_index = []

        for index, _ in enumerate(prediction_list):
            prediction_string = ''
            for list_index, content in enumerate(_):
                prediction_string += '{value} '.format(value=content)
            first_answer = prediction_string.rstrip().split('262 3280 318 ')[1].split(' ')[0]
            first_answer_index = len(prediction_string.split(first_answer)[0].rstrip().split(' '))
            last_answer = prediction_string.rstrip().split(' 50256 -100')[0].split(' ')[-1]
            last_answer_index = len(prediction_string.rstrip().split(' 50256 -100')[0].split(' '))
            answer_length = last_answer_index - first_answer_index
            answer4list_index = []
            for index in range(answer_length):
                answer4list_index += [_[first_answer_index + index]]
            answer_index += [answer4list_index]

        with torch.no_grad():
            weight_for_answer = []
            # weight_for_answer_tensor
            for index, _ in enumerate(answer_index):
                answer_length = len(answer_index[index])
                weight = []
                if answer_length == 1:
                    weight = [output_embedding[answer_index[index][0]]]

                elif answer_length > 1:
                    sum = torch.zeros([768], device=output_embedding.device)
                    for t in range(answer_length):
                        # print('answer_index[t][0]:{}\n'.format(answer_index[t][0]))
                        # print('answer_index[t]:{}\n'.format(answer_index[t]))
                        sum += output_embedding[answer_index[index][t]]

                    sum /= answer_length
                    weight = [sum]
                weight_for_answer += weight
                # weight_for_answer = weight_for_answer.unsqueeze(1)

            weight_for_answer_tensor = weight_for_answer[0].unsqueeze(0)
            for index, _ in enumerate(weight_for_answer):
                if index > 0:
                    weight_for_answer_tensor = torch.cat(
                        (weight_for_answer_tensor, weight_for_answer[index].unsqueeze(0)), dim=0)
            weight_for_answer_tensor = weight_for_answer_tensor.unsqueeze(1)
        #end = time.time()
        #print('upper_model time:{}\n'.format((end - start)))
        #start = time.time()
        lower_outputs = self.lower_model(
            up_hidden_states=weight_for_answer_tensor,
            input_ids=input_ques_exp_ids,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=ques_exp_segment_ids,
            position_ids=None,
            encoder_hidden_states=img_embeddings,
            encoder_attention_mask=None,
            labels=labels4ques_exp,
            use_cache=False,
            return_dict=True,
            test_or_train='train')

        answer_logits = upper_outputs.logits[0]
        #end = time.time()
        #print('lower_model time:{}\n'.format((end - start)))
        if reinforcement_flag == True:
            loss_ea = 0.
            loss_r = 0.
            batch_norm_ae = 0
            batch_norm_r = 0
            #re_start = time.time()
            for batch_index, label in zip(range(answer_logits.shape[0]), prediction_list):

                # prompt_ids
                prompt = tokenizer.decode(input_ques_exp_ids[batch_index]).split('<|endoftext|>')[0]
                tokenized_prompt = tokenizer.tokenize(prompt) + [tokenizer.bos_token] + tokenizer.tokenize(' because')
                prompt_len = len(tokenized_prompt)
                # padding_len = 39 - prompt_len
                # tokenized_prompt = tokenized_prompt + [tokenizer.pad_token] * padding_len
                tokenized_prompt_ids = tokenizer.convert_tokens_to_ids(tokenized_prompt)
                tokenized_prompt_ids = torch.tensor(tokenized_prompt_ids, dtype=torch.long).unsqueeze(0).to(device)

                # prompt_seg_ids
                q_segment_id, a_segment_id, e_segment_id = tokenizer.convert_tokens_to_ids(
                    ['<question>', '<answer>', '<explanation>'])
                question_prompt = tokenizer.tokenize(prompt)
                question_prompt_len = len(question_prompt)
                question_seg_ids = [q_segment_id] * question_prompt_len + [e_segment_id] * (
                            prompt_len - question_prompt_len)
                # question_seg_ids = [q_segment_id] * question_prompt_len + [e_segment_id] * (39 - question_prompt_len)
                question_seg_ids = torch.tensor(question_seg_ids, dtype=torch.long).unsqueeze(0).to(device)

                # answer_labels
                answer_labels = labels4ques_ans[batch_index].unsqueeze(0)

                # text_embedding = model.lower_model.transformer.wte(tokenized_prompt_ids)

                answer = answer_logits[batch_index].unsqueeze(0)

                label_string = ''
                for list_index, content in enumerate(label):
                    label_string += '{value} '.format(value=content)
                first_answer = label_string.rstrip().split('262 3280 318 ')[1].split(' ')[0]
                first_answer_index = len(label_string.split(first_answer)[0].rstrip().split(' '))
                last_answer = label_string.rstrip().split(' 50256 -100')[0].split(' ')[-1]
                last_answer_index = len(label_string.rstrip().split(' 50256 -100')[0].split(' '))
                answer = answer[:, first_answer_index:last_answer_index + 1, :]
                answer_label = label[first_answer_index:last_answer_index + 1]

                if 780 in input_ques_exp_ids[batch_index]:
                    gt_explantion = \
                    tokenizer.decode(input_ques_exp_ids[batch_index]).split('because')[1].split('<|endoftext|>')[0].lstrip()
                    gt_explantion_len = len(gt_explantion.split(' '))
                    identifier = 'yes'
                else:
                    identifier = 'no'

                beam_img_embeddings = img_embeddings[batch_index].unsqueeze(0)
                beam_up_hidden_states = weight_for_answer_tensor[batch_index].unsqueeze(0)
                #beam_start = time.time()
                beam_texts, scores = self.generate_beam(tokenizer, tokenized_prompt_ids, question_seg_ids,
                                                        beam_img_embeddings, beam_up_hidden_states, beam_size=2)
                #beam_end = time.time()
                #print('beam time:{}'.format((beam_end - beam_start)))
                if identifier == 'yes':
                    beam_texts.append(gt_explantion)

                for beam_index in range(len(beam_texts)):
                    beam_texts[beam_index] = ' ' + beam_texts[beam_index]

                for explanation_index in range(len(beam_texts)):
                    if explanation_index == 2 and identifier == 'no':
                        break

                    explanation_ids = torch.tensor(tokenizer(beam_texts[explanation_index])['input_ids'],
                                                   dtype=torch.long).to(device)
                    explanation_ids = torch.cat((explanation_ids, torch.tensor([50256], device=explanation_ids.device)))
                    explanation_len = explanation_ids.shape[0]
                    ques_exp_ids = torch.cat((tokenized_prompt_ids, explanation_ids.unsqueeze(0)), dim=1)

                    explanation_seg_ids = [e_segment_id] * explanation_len
                    explanation_seg_ids = torch.tensor(explanation_seg_ids, dtype=torch.long).unsqueeze(0).to(device)
                    ques_exp_seg_ids = torch.cat((question_seg_ids, explanation_seg_ids), dim=1)

                    answer_ids = torch.tensor(answer_label, dtype=torch.long).unsqueeze(0).to(device)
                    answer_ids = torch.cat((torch.tensor([[262,3280,318]], device=answer_ids.device), answer_ids), dim=1)
                    answer_seg_ids = [a_segment_id] * len(answer_ids[0,])
                    answer_seg_ids = torch.tensor(answer_seg_ids, dtype=torch.long).unsqueeze(0).to(device)
                    q_e_a_seg_ids = torch.cat((ques_exp_seg_ids, answer_seg_ids), dim=1)

                    input_embedding = torch.cat(
                        (self.upper_model.transformer.wte(ques_exp_ids), self.upper_model.transformer.wte(answer_ids)),
                        dim=1)
                    reward_logits = self.upper_model(
                        input_ids=None,
                        past_key_values=None,
                        attention_mask=None,
                        position_ids=None,
                        inputs_embeds=input_embedding,
                        token_type_ids=q_e_a_seg_ids,
                        encoder_hidden_states=beam_img_embeddings,
                        encoder_attention_mask=None,
                        labels=None,
                        use_cache=False,
                        return_dict=True,
                        test_or_train='train'
                    )
                    # Note that index is very important here.
                    reward_logits = reward_logits.logits[0][:,
                                    ques_exp_ids.shape[1] - 1: ques_exp_ids.shape[1] - 1 + answer_ids.shape[1]]
                    reward_logits_softmax = F.softmax(reward_logits, dim=-1)
                    base_logits_softmax = F.softmax(answer, dim=-1)

                    gt_score = 0.
                    reward_score = 0.

                    for base_answer, index, reward_answer in zip(base_logits_softmax.squeeze(), answer_ids.squeeze(),
                                                                 reward_logits_softmax.squeeze()):
                        gt_score += base_answer[index]
                        reward_score += reward_answer[index]

                    gt_score = gt_score / reward_logits.shape[0]
                    reward_score = reward_score / reward_logits.shape[0]

                    if explanation_index == 2:
                        loss_ea = F.cross_entropy(reward_logits.squeeze(), answer_ids.squeeze()) + loss_ea
                        batch_norm_ae += 1
                    else:
                        # print('\nscores:{}\n'.format(scores[explanation_index]))
                        loss_r = torch.clamp((reward_score - gt_score) * 10 * scores[explanation_index], min=0.) + loss_r
                        batch_norm_r += 1
                        if (reward_score - gt_score) < 0:
                            loss_r = F.cross_entropy(reward_logits.squeeze(), answer_ids.squeeze()) + loss_r
                            batch_norm_r += 1
            #re_end = time.time()
            #print('reinforcement time:{}'.format((re_end - re_start)))

            loss_ea = loss_ea / (batch_norm_ae + 1e-8)
            loss_r = loss_r / (batch_norm_r + 1e-8)

        else:
            #start = time.time()
            loss_ea = 0.
            batch_norm_ae = 0
            for batch_index, label in zip(range(answer_logits.shape[0]), prediction_list):

                # prompt_ids
                prompt = tokenizer.decode(input_ques_exp_ids[batch_index]).split('<|endoftext|>')[0]
                tokenized_prompt = tokenizer.tokenize(prompt) + [tokenizer.bos_token] + tokenizer.tokenize(' because')
                prompt_len = len(tokenized_prompt)
                # padding_len = 39 - prompt_len
                # tokenized_prompt = tokenized_prompt + [tokenizer.pad_token] * padding_len
                tokenized_prompt_ids = tokenizer.convert_tokens_to_ids(tokenized_prompt)
                tokenized_prompt_ids = torch.tensor(tokenized_prompt_ids, dtype=torch.long).unsqueeze(0).to(device)

                # prompt_seg_ids
                q_segment_id, a_segment_id, e_segment_id = tokenizer.convert_tokens_to_ids(
                    ['<question>', '<answer>', '<explanation>'])
                question_prompt = tokenizer.tokenize(prompt)
                question_prompt_len = len(question_prompt)
                question_seg_ids = [q_segment_id] * question_prompt_len + [e_segment_id] * (
                        prompt_len - question_prompt_len)
                # question_seg_ids = [q_segment_id] * question_prompt_len + [e_segment_id] * (39 - question_prompt_len)
                question_seg_ids = torch.tensor(question_seg_ids, dtype=torch.long).unsqueeze(0).to(device)

                # answer_labels
                answer_labels = labels4ques_ans[batch_index].unsqueeze(0)

                # text_embedding = model.lower_model.transformer.wte(tokenized_prompt_ids)

                answer = answer_logits[batch_index].unsqueeze(0)

                label_string = ''
                for list_index, content in enumerate(label):
                    label_string += '{value} '.format(value=content)
                first_answer = label_string.rstrip().split('262 3280 318 ')[1].split(' ')[0]
                first_answer_index = len(label_string.split(first_answer)[0].rstrip().split(' '))
                last_answer = label_string.rstrip().split(' 50256 -100')[0].split(' ')[-1]
                last_answer_index = len(label_string.rstrip().split(' 50256 -100')[0].split(' '))
                answer = answer[:, first_answer_index:last_answer_index + 1, :]
                answer_label = label[first_answer_index:last_answer_index + 1]

                if 780 in input_ques_exp_ids[batch_index]:
                    gt_explantion = \
                        tokenizer.decode(input_ques_exp_ids[batch_index]).split('because')[1].split('<|endoftext|>')[
                            0].lstrip()
                    gt_explantion_len = len(gt_explantion.split(' '))
                    identifier = 'yes'
                else:
                    identifier = 'no'

                beam_img_embeddings = img_embeddings[batch_index].unsqueeze(0)

                # beam_end = time.time()
                # print('beam time:{}'.format((beam_end - beam_start)))
                if identifier == 'yes':
                    beam_texts = []
                    beam_texts.append(gt_explantion)
                    beam_texts[0] = ' ' + beam_texts[0]

                    explanation_ids = torch.tensor(tokenizer(beam_texts[0])['input_ids'],
                                                   dtype=torch.long).to(device)
                    explanation_ids = torch.cat((explanation_ids, torch.tensor([50256], device=explanation_ids.device)), dim=0)
                    explanation_len = explanation_ids.shape[0]
                    ques_exp_ids = torch.cat((tokenized_prompt_ids, explanation_ids.unsqueeze(0)), dim=1)

                    explanation_seg_ids = [e_segment_id] * explanation_len
                    explanation_seg_ids = torch.tensor(explanation_seg_ids, dtype=torch.long).unsqueeze(0).to(device)
                    ques_exp_seg_ids = torch.cat((question_seg_ids, explanation_seg_ids), dim=1)

                    answer_ids = torch.tensor(answer_label, dtype=torch.long).unsqueeze(0).to(device)
                    answer_ids = torch.cat((torch.tensor([[262,3280,318]], device=answer_ids.device), answer_ids), dim=1)
                    answer_seg_ids = [a_segment_id] * len(answer_ids[0,])
                    answer_seg_ids = torch.tensor(answer_seg_ids, dtype=torch.long).unsqueeze(0).to(device)
                    q_e_a_seg_ids = torch.cat((ques_exp_seg_ids, answer_seg_ids), dim=1)
                    input_ids = torch.cat((ques_exp_ids, answer_ids), dim=1)
                    #input_embedding = torch.cat(
                    #    (self.upper_model.transformer.wte(ques_exp_ids), self.upper_model.transformer.wte(answer_ids)),
                    #    dim=1)
                    reward_logits = self.upper_model(
                        input_ids=input_ids,
                        past_key_values=None,
                        attention_mask=None,
                        position_ids=None,
                        inputs_embeds=None,
                        token_type_ids=q_e_a_seg_ids,
                        encoder_hidden_states=beam_img_embeddings,
                        encoder_attention_mask=None,
                        labels=None,
                        use_cache=False,
                        return_dict=True,
                        test_or_train='train'
                    )
                    # Note that index is very important here.
                    reward_logits = reward_logits.logits[0][:,
                                    ques_exp_ids.shape[1] - 1: ques_exp_ids.shape[1] - 1 + answer_ids.shape[1]]

                    loss_ea = F.cross_entropy(reward_logits.squeeze(), answer_ids.squeeze()) + loss_ea
                    batch_norm_ae += 1


            loss_ea = loss_ea / (batch_norm_ae + 1e-8)


        #end = time.time()
        #if accelerator.is_main_process:
        #    print('time:{}\n'.format((end-start)))
        # loss = upper_outputs.loss[0] + lower_outputs.loss[0] + loss_ea + loss_r
        if reinforcement_flag == True:
            loss = upper_outputs.loss[0] + lower_outputs.loss[0] + loss_ea + loss_r
            return upper_outputs.loss[0], lower_outputs.loss[0], loss_ea, loss_r, loss
        else:
            loss = upper_outputs.loss[0] + lower_outputs.loss[0] + loss_ea
            return upper_outputs.loss[0], lower_outputs.loss[0], loss_ea, loss

        # return upper_outputs.loss[0], lower_outputs.loss[0], loss_ea, loss_r, loss


def sample_sequences(model, tokenizer, loader, epoch):
    model.eval()
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    max_len = 20

    #if epoch == 5 or epoch == 10 or epoch == 15:
    for i, batch in enumerate(loader):

        current_output_ans = []
        current_base_output_ans = []
        current_reward_output_ans = []
        current_output_exp = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ques_ans_ids, input_ques_exp_ids, ques_ans_segment_ids, ques_exp_segment_ids = batch
        img_embeddings = image_encoder(img)

        with torch.no_grad():

            for step in range(max_len + 1):

                if step == max_len:
                    break

                outputs = model.upper_model(input_ids=input_ques_ans_ids,
                                            past_key_values=None,
                                            attention_mask=None,
                                            token_type_ids=ques_ans_segment_ids,
                                            position_ids=None,
                                            encoder_hidden_states=img_embeddings,
                                            encoder_attention_mask=None,
                                            labels=None,
                                            use_cache=False,
                                            return_dict=True,
                                            test_or_train='test')

                hidden_states = outputs.hidden_states[0]

                base_ans_logits = outputs.logits[0]
                base_ans_logits = base_ans_logits[0, -1, :] / temperature
                base_ans_logits = top_filtering(base_ans_logits, top_k=top_k, top_p=top_p)
                base_ans_probs = F.softmax(base_ans_logits, dim=-1)
                base_ans_prev = torch.topk(base_ans_probs, 1)[1] if no_sample else torch.multinomial(base_ans_probs, 1)

                if base_ans_prev.item() in special_tokens_ids:
                    break

                new_ans_segment = special_tokens_ids[-2]

                new_ans_segment = torch.LongTensor([new_ans_segment]).to(device)
                current_base_output_ans.append(base_ans_prev.item())
                input_ques_ans_ids = torch.cat((input_ques_ans_ids, base_ans_prev.unsqueeze(0)), dim=1)
                ques_ans_segment_ids = torch.cat((ques_ans_segment_ids, new_ans_segment.unsqueeze(0)), dim=1)

            output_embedding = outputs.output_embedding.weight

            weight_for_answer = torch.zeros([hidden_states[0].shape[0], hidden_states[0].shape[2]])

            prediction_list = (input_ques_ans_ids.cpu().detach().numpy().tolist())[0]

            prediction_string = ''
            for index, _ in enumerate(prediction_list):
                prediction_string += '{value} '.format(value=_)

            all_answer = prediction_string.rstrip().split('262 3280 318')[1].lstrip()
            if len(all_answer) > 0:
                first_answer = all_answer.split(' ')[0]
                first_answer_index = len(prediction_string.split(first_answer)[0].rstrip().split(' '))
                answer_length = len(prediction_list) - first_answer_index
                answer_index = []
                for index in range(answer_length):
                    answer_index += [prediction_list[first_answer_index + index]]

                with torch.no_grad():
                    weight = torch.zeros([1, 768], device=output_embedding.device)
                    if answer_length == 1:
                        weight += output_embedding[answer_index[0]]
                        weight_for_answer = weight
                    else:
                        for t in range(answer_length):
                            weight += output_embedding[answer_index[t]]
                        weight = weight / answer_length
                        weight_for_answer = weight
                    weight_for_answer = weight_for_answer.unsqueeze(1)

            else:
                weight_for_answer = torch.zeros([hidden_states[0].shape[0], hidden_states[0].shape[2]]).unsqueeze(1)

            answer_logits = outputs.logits[0]
            for batch_index in range(answer_logits.shape[0]):

                # prompt_ids
                prompt = tokenizer.decode(input_ques_exp_ids[batch_index]).split('<|endoftext|>')[0]
                tokenized_prompt = tokenizer.tokenize(prompt) + [tokenizer.bos_token] + tokenizer.tokenize(' because')
                prompt_len = len(tokenized_prompt)

                tokenized_prompt_ids = tokenizer.convert_tokens_to_ids(tokenized_prompt)
                tokenized_prompt_ids = torch.tensor(tokenized_prompt_ids, dtype=torch.long).unsqueeze(0).to(device)

                # prompt_seg_ids
                q_segment_id, a_segment_id, e_segment_id = tokenizer.convert_tokens_to_ids(
                    ['<question>', '<answer>', '<explanation>'])
                question_prompt = tokenizer.tokenize(prompt)
                question_prompt_len = len(question_prompt)
                question_seg_ids = [q_segment_id] * question_prompt_len + [e_segment_id] * (
                        prompt_len - question_prompt_len)

                question_seg_ids = torch.tensor(question_seg_ids, dtype=torch.long).unsqueeze(0).to(device)

                beam_img_embeddings = img_embeddings[batch_index].unsqueeze(0)
                beam_up_hidden_states = weight_for_answer[batch_index].unsqueeze(0)

                beam_texts, scores = model.generate_beam(tokenizer, tokenized_prompt_ids, question_seg_ids,
                                                         beam_img_embeddings, beam_up_hidden_states, beam_size=2, max_len = 20)
                exp_beam_texts = beam_texts

                order = scores.argsort(descending=True)
                beam_texts = [beam_texts[i] for i in order]
                beam_texts = [beam_texts[0]]
                for beam_index in range(len(beam_texts)):
                    if len(beam_texts[beam_index]) == 0:
                        _ = 1
                    else:
                        beam_texts[beam_index] = ' ' + beam_texts[beam_index]



                for explanation_index in range(len(beam_texts)):
                    explanation_ids = torch.tensor(tokenizer(beam_texts[explanation_index])['input_ids'],
                                                   dtype=torch.long).to(device)
                    explanation_ids = torch.cat((explanation_ids, torch.tensor([50256],device=explanation_ids.device)),dim=0)
                    explanation_len = explanation_ids.shape[0]
                    ques_exp_ids = torch.cat((tokenized_prompt_ids, explanation_ids.unsqueeze(0)), dim=1)

                    explanation_seg_ids = [e_segment_id] * explanation_len
                    explanation_seg_ids = torch.tensor(explanation_seg_ids, dtype=torch.long).unsqueeze(0).to(device)
                    ques_exp_seg_ids = torch.cat((question_seg_ids, explanation_seg_ids), dim=1)

                    #answer_label = current_output_ans  # test don't have gt, so use answer index produced by answer module above.
                    #answer_ids = torch.tensor(answer_label, dtype=torch.long).unsqueeze(0).to(device)
                    answer_ids = torch.tensor([262,3280,318], device=ques_exp_seg_ids.device).unsqueeze(0)
                    input_ques_exp_ans_ids = torch.cat((ques_exp_ids, answer_ids), dim=1)

                    answer_seg_ids = [a_segment_id] * len(answer_ids[0,])
                    answer_seg_ids = torch.tensor(answer_seg_ids, dtype=torch.long).unsqueeze(0).to(device)
                    q_e_a_seg_ids = torch.cat((ques_exp_seg_ids, answer_seg_ids), dim=1)

                    #answer_beam_texts, answer_scores = generate_beam_val(model, tokenizer, ques_exp_ids, answer_ids,
                    #                                                     q_e_a_seg_ids, beam_img_embeddings)

                    for step in range(max_len + 1):

                        if step == max_len:
                            break

                        outputs = model.upper_model(input_ids=input_ques_exp_ans_ids,
                                                    past_key_values=None,
                                                    attention_mask=None,
                                                    token_type_ids=q_e_a_seg_ids,
                                                    position_ids=None,
                                                    encoder_hidden_states=beam_img_embeddings,
                                                    encoder_attention_mask=None,
                                                    labels=None,
                                                    use_cache=False,
                                                    return_dict=True,
                                                    test_or_train='test')

                        hidden_states = outputs.hidden_states[0]

                        reward_ans_logits = outputs.logits[0]
                        reward_ans_logits = reward_ans_logits[0, -1, :] / temperature
                        reward_ans_logits = top_filtering(reward_ans_logits, top_k=top_k, top_p=top_p)
                        reward_ans_probs = F.softmax(reward_ans_logits, dim=-1)
                        reward_ans_prev = torch.topk(reward_ans_probs, 1)[1] if no_sample else torch.multinomial(
                            reward_ans_probs, 1)

                        if reward_ans_prev.item() in special_tokens_ids:
                            break

                        new_ans_segment = special_tokens_ids[-2]

                        new_ans_segment = torch.LongTensor([new_ans_segment]).to(device)
                        current_reward_output_ans.append(reward_ans_prev.item())
                        input_ques_exp_ans_ids = torch.cat((input_ques_exp_ans_ids, reward_ans_prev.unsqueeze(0)), dim=1)
                        q_e_a_seg_ids = torch.cat((q_e_a_seg_ids, new_ans_segment.unsqueeze(0)),
                                                         dim=1)


        #if answer_scores[0] < answer_scores[1]:
        #    current_output_ans = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answer_beam_texts[1]))
        #else:
        #    current_output_ans = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answer_beam_texts[0]))
        current_output_ans = current_reward_output_ans


        if scores[0] < scores[1]:
            current_output_exp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(exp_beam_texts[1]))
        else:
            current_output_exp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(exp_beam_texts[0]))

        decoded_ans = tokenizer.decode(current_output_ans, skip_special_tokens=True).lstrip()
        decoded_exp = tokenizer.decode(current_output_exp, skip_special_tokens=True).lstrip()

        results_full.append({"image_id": img_id.item(), "caption": decoded_ans + ' because ' + decoded_exp})

        results_exp.append({"image_id": img_id.item(), "caption": decoded_exp})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')
    '''
    else:
        for i, batch in enumerate(loader):

            current_output_ans = []
            current_output_exp = []
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, img_id, input_ques_ans_ids, input_ques_exp_ids, ques_ans_segment_ids, ques_exp_segment_ids = batch
            img_embeddings = image_encoder(img)

            with torch.no_grad():

                for step in range(max_len + 1):

                    if step == max_len:
                        break

                    outputs = model.upper_model(input_ids=input_ques_ans_ids,
                                                past_key_values=None,
                                                attention_mask=None,
                                                token_type_ids=ques_ans_segment_ids,
                                                position_ids=None,
                                                encoder_hidden_states=img_embeddings,
                                                encoder_attention_mask=None,
                                                labels=None,
                                                use_cache=False,
                                                return_dict=True,
                                                test_or_train='test')

                    hidden_states = outputs.hidden_states[0]

                    base_ans_logits = outputs.logits[0]
                    base_ans_logits = base_ans_logits[0, -1, :] / temperature
                    base_ans_logits = top_filtering(base_ans_logits, top_k=top_k, top_p=top_p)
                    base_ans_probs = F.softmax(base_ans_logits, dim=-1)
                    base_ans_prev = torch.topk(base_ans_probs, 1)[1] if no_sample else torch.multinomial(base_ans_probs,
                                                                                                         1)

                    if base_ans_prev.item() in special_tokens_ids:
                        break

                    new_ans_segment = special_tokens_ids[-2]

                    new_ans_segment = torch.LongTensor([new_ans_segment]).to(device)
                    current_output_ans.append(base_ans_prev.item())
                    input_ques_ans_ids = torch.cat((input_ques_ans_ids, base_ans_prev.unsqueeze(0)), dim=1)
                    ques_ans_segment_ids = torch.cat((ques_ans_segment_ids, new_ans_segment.unsqueeze(0)), dim=1)

                output_embedding = outputs.output_embedding.weight

                weight_for_answer = torch.zeros([hidden_states[0].shape[0], hidden_states[0].shape[2]])

                prediction_list = (input_ques_ans_ids.cpu().detach().numpy().tolist())[0]

                prediction_string = ''
                for index, _ in enumerate(prediction_list):
                    prediction_string += '{value} '.format(value=_)

                all_answer = prediction_string.rstrip().split('262 3280 318')[1].lstrip()
                if len(all_answer) > 0:
                    first_answer = all_answer.split(' ')[0]
                    first_answer_index = len(prediction_string.split(first_answer)[0].rstrip().split(' '))
                    answer_length = len(prediction_list) - first_answer_index
                    answer_index = []
                    for index in range(answer_length):
                        answer_index += [prediction_list[first_answer_index + index]]

                    with torch.no_grad():
                        weight = torch.zeros([1, 768], device=output_embedding.device)
                        if answer_length == 1:
                            weight += output_embedding[answer_index[0]]
                            weight_for_answer = weight
                        else:
                            for t in range(answer_length):
                                weight += output_embedding[answer_index[t]]
                            weight = weight / answer_length
                            weight_for_answer = weight
                        weight_for_answer = weight_for_answer.unsqueeze(1)

                else:
                    weight_for_answer = torch.zeros([hidden_states[0].shape[0], hidden_states[0].shape[2]]).unsqueeze(1)

                # greedy generate exp.

                for step in range(max_len + 1):
                    if step == max_len:
                        break


                    outputs = model.lower_model(
                        up_hidden_states=weight_for_answer,
                        input_ids=input_ques_exp_ids,
                        past_key_values=None,
                        attention_mask=None,
                        token_type_ids=ques_exp_segment_ids,
                        position_ids=None,
                        encoder_hidden_states=img_embeddings,
                        encoder_attention_mask=None,
                        labels=None,
                        use_cache=False,
                        return_dict=True,
                        test_or_train='test',
                        step=step)

                    exp_logits = outputs.logits[0]
                    exp_logits = exp_logits[0, -1, :] / temperature
                    exp_logits = top_filtering(exp_logits, top_k=top_k, top_p=top_p)
                    exp_probs = F.softmax(exp_logits, dim=-1)
                    exp_prev = torch.topk(exp_probs, 1)[1] if no_sample else torch.multinomial(exp_probs, 1)

                    if exp_prev.item() in special_tokens_ids:
                        break

                    new_exp_segment = special_tokens_ids[-1]

                    new_exp_segment = torch.LongTensor([new_exp_segment]).to(device)
                    current_output_exp.append(exp_prev.item())
                    input_ques_exp_ids = torch.cat((input_ques_exp_ids, exp_prev.unsqueeze(0)), dim=1)
                    ques_exp_segment_ids = torch.cat((ques_exp_segment_ids, new_exp_segment.unsqueeze(0)), dim=1)

            decoded_ans = tokenizer.decode(current_output_ans, skip_special_tokens=True).lstrip()
            decoded_exp = tokenizer.decode(current_output_exp, skip_special_tokens=True).lstrip()

            results_full.append({"image_id": img_id.item(), "caption": decoded_ans + ' because ' + decoded_exp})

            results_exp.append({"image_id": img_id.item(), "caption": decoded_exp})
            print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')

    '''
    return results_full, results_exp


def generate_beam_val(model, tokenizer, ques_exp_ids, answer_ids, q_e_a_seg_ids, beam_img_embeddings):
    beam_size = 2
    beam_text_1 = []
    beam_text_2 = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    new_exp_segment = special_tokens_ids[-1]
    new_exp_segment = torch.LongTensor([new_exp_segment]).to(device)
    max_len = 5
    scores = None
    tokens = None

    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    scores_list = None

    # text_embedding = model.upper_model.transformer.wte(prompt_ids)
    # input_embedding = text_embedding
    input_embedding = torch.cat(
        (model.upper_model.transformer.wte(ques_exp_ids), model.upper_model.transformer.wte(answer_ids)), dim=1)

    for step in range(max_len + 1):
        if step == max_len:
            break

        outputs = model.upper_model(
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=q_e_a_seg_ids,
            position_ids=None,
            inputs_embeds=input_embedding,
            encoder_hidden_states=beam_img_embeddings,
            encoder_attention_mask=None,
            labels=None,
            use_cache=False,
            return_dict=True,
            test_or_train='test',
            step=step
        )

        hidden_states = outputs.hidden_states[0]
        beam_logits = outputs.logits[0]
        beam_logits = beam_logits[:, -1, :] / temperature
        beam_logits = beam_logits.softmax(-1) + 1e-16
        beam_logits = beam_logits.log()
        # beam_logits_exp = torch.exp(beam_logits)
        # partition = beam_logits_exp.sum(dim=-1, keepdim=True)
        # beam_logits = beam_logits_exp / partition
        # beam_logits = torch.clamp(beam_logits, min=-50)
        if scores is None:
            scores, next_token = beam_logits.topk(beam_size, -1)
            # scores = scores.log()
            input_embedding = input_embedding.expand(beam_size, *input_embedding.shape[1:])
            q_e_a_seg_ids = q_e_a_seg_ids.expand(beam_size, *q_e_a_seg_ids.shape[1:])
            # img_embeddings = img_embeddings.expand(beam_size, *img_embeddings.shape[1:])
            new_exp_segment = new_exp_segment.expand(beam_size, *new_exp_segment.shape[1:])
            next_token, scores = next_token.permute(1, 0), scores.squeeze(0)
            scores_list = scores.unsqueeze(-1)
            if tokens is None:
                tokens = next_token

            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_token), dim=1)

        else:
            beam_logits[is_stopped] = -float(np.inf)
            beam_logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + beam_logits
            # print('\nscores[:None]:{}\n'.format(scores[:,None]))
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_token = scores_sum_average.view(-1).topk(beam_size, -1)
            next_token_source = next_token // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_token_source]
            next_token = next_token % scores_sum.shape[1]
            # if torch.isnan(beam_logits[next_token_source, next_token]).any():
            #    _ = 1
            # else:
            scores_list = torch.cat(
                [scores_list[next_token_source], beam_logits[next_token_source, next_token].unsqueeze(-1)], dim=-1)
            next_token = next_token.unsqueeze(1)
            tokens = tokens[next_token_source]
            tokens = torch.cat((tokens, next_token), dim=1)
            input_embedding = input_embedding[next_token_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_token_source]

        next_token_embedding = model.upper_model.transformer.wte(next_token.squeeze()).view(
            input_embedding.shape[0], 1, -1)
        input_embedding = torch.cat((input_embedding, next_token_embedding), dim=1)
        q_e_a_seg_ids = torch.cat((q_e_a_seg_ids, new_exp_segment.unsqueeze(1)), dim=1)

        # is_stopped = is_stopped + next_token.eq(special_tokens_ids[0]).squeeze()
        for i in range(2):
            is_stopped[i] = is_stopped[i] + torch.tensor(next_token[i].item() in special_tokens_ids)
        if is_stopped.all():
            break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length) - 1]).strip()
        for output, length in zip(output_list, seq_lengths)
    ]
    # token = []
    # token.append(torch.tensor(tokenizer(output_texts[0])['input_ids'], dtype=torch.int64).to(device))
    scores = -torch.mean(scores_list[0, :tokens.shape[1]]).unsqueeze(0)
    for i in range(1, beam_size):
        # token.append(torch.tensor(tokenizer(output_texts[i])['input_ids'], dtype=torch.int64).to(device))
        scores = torch.cat((scores, -torch.mean(scores_list[i, :tokens.shape[1]]).unsqueeze(0)), dim=0)

    return output_texts, scores


def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.upper_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.upper_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.lower_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.lower_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
accelerator = Accelerator()
device = accelerator.device

finetune_pretrained = False  # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
mask_size = 14
ckpt_path = 'ckpts/'
caption_save_path = 'cococaption/results/'
annFileExp = '/home/intership/nlxgpt/cococaption/annotations/vqaX_test_annot_exp.json'
annFileFull = '/home/intership/nlxgpt/cococaption/annotations/vqaX_test_annot_full.json'
nle_data_train_path = 'train_HX_data.json'
nle_data_test_path = '/home/intership/nlxgpt/nle_data/VQA-X/vqaX_test.json'
nle_data_val_path = '/home/intership/nlxgpt/nle_data/VQA-X/vqaX_val.json'
vqahat_test_data_path = 'val_data_with_answer.json'
max_seq_len = 40
max_exp_seq_len = 40
load_from_epoch = None
no_sample = True
top_k = 0
top_p = 0.9
batch_size = 32  # per GPU
num_train_epochs = 20
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
gradient_accumulation_steps = 1
start_epoch = 0
temperature = 1

image_encoder = ImageEncoder(device).to(device)
change_requires_grad(image_encoder, False)

if load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, load_from_epoch)

else:

    if finetune_pretrained:
        tokenizer, model = load_pretrained()
        optimizer = get_optimizer(model, learning_rate)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('ckpts/nle_gpt2_tokenizer_0', cache_dir='ckpts/nle_gpt2_tokenizer_0')
        orig_num_tokens = len(tokenizer.encoder)

        num_new_tokens = tokenizer.add_special_tokens(
            {'pad_token': '<pad>', 'additional_special_tokens': ['<question>', '<answer>', '<explanation>']})

        # assert len(tokenizer) == orig_num_tokens + num_new_tokens

        model = NLXmodel()

        model.upper_model.resize_token_embeddings(len(tokenizer))
        model.lower_model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)

print("Model Setup Ready...")

img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = VQAXTrainDataset(path=nle_data_train_path,
                                 transform=img_transform,
                                 tokenizer=tokenizer,
                                 max_seq_len=max_seq_len,
                                 max_exp_seq_len=max_exp_seq_len
                                 )

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True)

train_loader_r = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=8,
                                             shuffle=True,
                                             pin_memory=True)

test_dataset = VQAXEvalDataset(path=nle_data_test_path,
                               transform=img_transform,
                               tokenizer=tokenizer,
                               max_seq_len=max_seq_len)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = 0  # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)

for epoch in range(start_epoch, num_train_epochs):

    model.train()
    accum_loss = 0
    accum_up_loss = 0
    accum_low_loss = 0
    accum_ae_loss = 0
    accum_r_loss = 0
    if epoch == 5 or epoch == 10 or epoch == 15:
        for step, batch in enumerate(train_loader_r):
            if step == 1000:
                break
            #with torch.autograd.set_detect_anomaly(True):
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, _, input_ques_ans_ids, input_ques_exp_ids, labels4ques_ans, \
                labels4ques_exp, ques_ans_segment_ids, ques_exp_segment_ids = batch

            img_embeddings = image_encoder(img)


            up_loss, low_loss, loss_ea, loss_r, loss = model(img_embeddings=img_embeddings,
                                                             input_ques_ans_ids=input_ques_ans_ids,
                                                             input_ques_exp_ids=input_ques_exp_ids,
                                                             labels4ques_ans=labels4ques_ans,
                                                             labels4ques_exp=labels4ques_exp,
                                                             ques_ans_segment_ids=ques_ans_segment_ids,
                                                             ques_exp_segment_ids=ques_exp_segment_ids,
                                                             test_or_train='train',
                                                             reinforcement_flag=True)
            up_loss = up_loss / gradient_accumulation_steps
            low_loss = low_loss / gradient_accumulation_steps
            loss_ea = loss_ea / gradient_accumulation_steps
            loss_r = loss_r / gradient_accumulation_steps
            loss = loss / gradient_accumulation_steps

            accelerator.backward(loss)

            accum_up_loss += up_loss.item()
            accum_low_loss += low_loss.item()
            if type(loss_ea) == float:
                _ = 1
            else:
                accum_ae_loss += loss_ea.item()
            if type(loss_r) == float:
                _ = 1
            else:
                accum_r_loss += loss_r.item()
            accum_loss += loss.item()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accelerator.print(
                    "\rEpoch {} / {}, Iter {} / {}, Up_Loss: {:.3f}, Low_Loss: {:.3f} , AE_Loss: {:.3f}, R_Loss: {:.3f}, Loss: {:.3f}".format(
                        epoch, num_train_epochs, step, len(train_loader_r),
                        accum_up_loss, accum_low_loss, accum_ae_loss, accum_r_loss, accum_loss), end='          ')
                accum_up_loss = 0
                accum_low_loss = 0
                accum_ae_loss = 0
                accum_r_loss = 0
                accum_loss = 0
    else:
        for step, batch in enumerate(train_loader):
            #with torch.autograd.set_detect_anomaly(True):
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, _, input_ques_ans_ids, input_ques_exp_ids, labels4ques_ans, \
                labels4ques_exp, ques_ans_segment_ids, ques_exp_segment_ids = batch

            img_embeddings = image_encoder(img)


            up_loss, low_loss, loss_ea, loss = model(img_embeddings=img_embeddings,
                                                             input_ques_ans_ids=input_ques_ans_ids,
                                                             input_ques_exp_ids=input_ques_exp_ids,
                                                             labels4ques_ans=labels4ques_ans,
                                                             labels4ques_exp=labels4ques_exp,
                                                             ques_ans_segment_ids=ques_ans_segment_ids,
                                                             ques_exp_segment_ids=ques_exp_segment_ids,
                                                             test_or_train='train')
            up_loss = up_loss / gradient_accumulation_steps
            low_loss = low_loss / gradient_accumulation_steps
            ea_loss = loss_ea / gradient_accumulation_steps
            loss = loss / gradient_accumulation_steps

            accelerator.backward(loss)

            accum_up_loss += up_loss.item()
            accum_low_loss += low_loss.item()
            if type(loss_ea) == float:
                _ = 1
            else:
                accum_ae_loss += loss_ea.item()
            accum_loss += loss.item()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accelerator.print(
                    "\rEpoch {} / {}, Iter {} / {}, Up_Loss: {:.3f}, Low_Loss: {:.3f}, EA_Loss: {:.3f}, Loss: {:.3f}".format(
                        epoch, num_train_epochs, step, len(train_loader),
                        accum_up_loss, accum_low_loss, accum_ae_loss, accum_loss), end='          ')
                accum_up_loss = 0
                accum_low_loss = 0
                accum_ae_loss = 0
                accum_loss = 0

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)

    if accelerator.is_main_process:
        results_full, results_exp = sample_sequences(unwrapped_model, tokenizer, test_loader, epoch)

        resFileExp = caption_save_path + 'captions_exp_' + str(epoch) + '.json'
        unf_resFileExp = caption_save_path + 'unf_captions_exp_' + str(epoch) + '.json'
        unf_resFileFull = caption_save_path + 'unf_captions_full_' + str(epoch) + '.json'
        save_scores_pathExp = caption_save_path + 'scores_exp_' + str(epoch) + '.json'

        with open(unf_resFileExp, 'w') as w:
            json.dump(results_exp, w)

        with open(unf_resFileFull, 'w') as w:
            json.dump(results_full, w)

        '''
        #with open(unf_resFileExp) as f:
            results_exp = json.load(f)

        with open(unf_resFileFull) as f:
            results_full = json.load(f)
        '''

        # unfiltered results
        # get_scores(annFileExp, unf_resFileExp, save_scores_pathExp)

        # filtered results
        filter_and_get_scores(resFileExp, save_scores_pathExp, results_full, results_exp)
