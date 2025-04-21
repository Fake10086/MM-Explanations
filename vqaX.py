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


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    #model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    model = NLXmodel()
    # model = GPT2LMHeadModel.from_pretrained('/home/lizhengyi/2022/task_basecalling/ex_try/bonito_copy/distillgpt2', config = config)
    #model = opt['model']
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
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)   # load model with config
    return tokenizer, model
    

def save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
    if epoch % 5 == 0:
        unwrapped_model.save_pretrained(ckpt_path, epoch)
        
        opt = {'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                    **kwargs}
    
        accelerator.save(opt, ckpt_path + filename)
    
# def get_scores(annFile, resFile, save_scores_path):
    
#     coco = COCO(annFile)
#     cocoRes = coco.loadRes(resFile)
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     cocoEval.evaluate()
#     with open(save_scores_path, 'w') as w:
#         json.dump(cocoEval.eval, w)
    
    
def filter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predictions):

    all_file = json.load(open(nle_data_test_path, 'r'))
    
    gt_answers = {}
    for key,value in all_file.items():
        gt_answers[int(key)] = data_utils.proc_ans(value['answers'])
        
    pred_answers = {}
    for item in full_predictions:
        pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()
        
    correct_keys = []
    for key,value in pred_answers.items():
        gt_answer = gt_answers[key]
        # to measure accuracy for VQA, please change "==" to "in" (if value in gt_answer:)
        # you need to also change the proc_ans funtion in utils/data_uitls.py to return: list(ans_prob_dict.keys())
        if value == gt_answer:
            correct_keys.append(key)
            

    exp_preds = [item for item in exp_predictions if item['image_id'] in correct_keys]

    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)
        
    coco = COCO(annFileExp)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    
    with open(save_scores_pathExp, 'w') as w:
        json.dump(cocoEval.eval, w)
    

class VQAXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len, max_exp_seq_len):
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.max_exp_seq_len = max_exp_seq_len
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        
        for k,v in self.data.items():   
            if len(v['explanation']) > 1:   # some questions have more than one explanation
                # duplicate them for loading. -1 because one explanation is already in ids_list
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)    

        self.index_tracker = {k: len(v['explanation']) - 1 for k,v in self.data.items()}
        

    def __getitem__(self, i):
        
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = data_utils.proc_ques(sample['question'])    # question
        answer = data_utils.proc_ans(sample['answers'])

        exp_idx = self.index_tracker[quention_id]    # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[quention_id] -= 1    # decrease usage
                
        exp_text = sample['explanation'][exp_idx]   # explanation
        
        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        ques_tokens = self.tokenizer.tokenize(text_a)
        ques_labels = [-100] * len(ques_tokens)   # we dont want to predict the question, set to pad to ignore in XE
        ques_segment_ids = [q_segment_id] * len(ques_tokens)

        # concat questions and answers for upper stream
        ans_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is " + answer) + [self.tokenizer.eos_token]
        answer_len = len(ans_tokens)
        tokens4ques_ans = ques_tokens + ans_tokens
        
        # concat questions and explanations for lower stream
        exp_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" because " + exp_text) + [self.tokenizer.eos_token]
        exp_len = len(exp_tokens)
        tokens4ques_exp = ques_tokens + exp_tokens
        
        # labels will be shifted in the model, so for now set them same as tokens
        labels4ques_ans = ques_labels + [-100] + ans_tokens[1:]
        labels4ques_exp = ques_labels + [-100] + exp_tokens[1:]   
        ques_ans_segment_ids = ques_segment_ids + [a_segment_id] * answer_len
        ques_exp_segment_ids = ques_segment_ids + [e_segment_id] * exp_len

        # for ques_ans output
        if len(tokens4ques_ans) > self.max_seq_len :
            tokens4ques_ans = tokens4ques_ans[:self.max_seq_len]
            labels4ques_ans = labels4ques_ans[:self.max_seq_len]
            ques_ans_segment_ids = ques_ans_segment_ids[:self.max_seq_len]

        assert len(tokens4ques_ans) == len(labels4ques_ans)
        assert len(tokens4ques_ans) == len(ques_ans_segment_ids)

        ques_ans_seq_len = len(tokens4ques_ans)
        padding_len = self.max_seq_len - ques_ans_seq_len
        tokens4ques_ans += ([self.tokenizer.pad_token] * padding_len)
        labels4ques_ans += ([-100] * padding_len)
        ques_ans_segment_ids += ([e_segment_id] * padding_len)

        input_ques_ans_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_ans)
        input_ques_ans_ids = torch.tensor(input_ques_ans_ids, dtype=torch.long)

        labels4ques_ans = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels4ques_ans]
        labels4ques_ans = torch.tensor(labels4ques_ans, dtype=torch.long)
        
        ques_ans_segment_ids = torch.tensor(ques_ans_segment_ids, dtype=torch.long)

        # for ques_exp output
        if len(tokens4ques_exp) > self.max_seq_len :
            tokens4ques_exp = tokens4ques_exp[:self.max_seq_len]
            labels4ques_exp = labels4ques_exp[:self.max_seq_len]
            ques_exp_segment_ids = ques_exp_segment_ids[:self.max_seq_len]

        assert len(tokens4ques_exp) == len(labels4ques_exp)
        assert len(tokens4ques_exp) == len(ques_exp_segment_ids)
        
        ques_exp_seq_len = len(tokens4ques_exp)
        padding_len = self.max_exp_seq_len - ques_exp_seq_len
        tokens4ques_exp += ([self.tokenizer.pad_token] * padding_len)
        labels4ques_exp += ([-100] * padding_len)
        ques_exp_segment_ids += ([e_segment_id] * padding_len)

        input_ques_exp_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_exp)
        input_ques_exp_ids = torch.tensor(input_ques_exp_ids, dtype=torch.long)

        labels4ques_exp = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels4ques_exp]
        labels4ques_exp = torch.tensor(labels4ques_exp, dtype=torch.long)
        
        ques_exp_segment_ids = torch.tensor(ques_exp_segment_ids, dtype=torch.long)
        
        folder = '/home/mj/nlxgpt/images/train2014/' if 'train' in img_name else '/home/mj/nlxgpt/images/val2014/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])
        
        return (img, qid, input_ques_ans_ids, input_ques_exp_ids, labels4ques_ans, labels4ques_exp, ques_ans_segment_ids, ques_exp_segment_ids)

    def __len__(self):
        return len(self.ids_list)
    
class VQAXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> because <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())


    def __getitem__(self, i):
        
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = data_utils.proc_ques(sample['question'])    # question

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        ques_tokens = self.tokenizer.tokenize(text_a)
        ques_segment_ids = [q_segment_id] * len(ques_tokens)

        ans_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is ")
        answer_len = len(ans_tokens)
        tokens4ques_ans = ques_tokens + ans_tokens

        ques_ans_segment_ids = ques_segment_ids + [a_segment_id] * answer_len

        input_ques_ans_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_ans)
        input_ques_ans_ids = torch.tensor(input_ques_ans_ids, dtype=torch.long)
        ques_ans_segment_ids = torch.tensor(ques_ans_segment_ids, dtype=torch.long)

        exp_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" because ")
        exp_len = len(exp_tokens)
        tokens4ques_exp = ques_tokens + exp_tokens

        ques_exp_segment_ids = ques_segment_ids + [e_segment_id] * exp_len

        input_ques_exp_ids = self.tokenizer.convert_tokens_to_ids(tokens4ques_exp)
        input_ques_exp_ids = torch.tensor(input_ques_exp_ids, dtype=torch.long)
        ques_exp_segment_ids = torch.tensor(ques_exp_segment_ids, dtype=torch.long)
        
        folder = '/home/mj/nlxgpt/images/train2014/' if 'train' in img_name else '/home/mj/nlxgpt/images/val2014/'   # test and val are both in val2014
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
        #ckpt_path + model_name
        self.upper_model = GPT2LMHeadModel.from_pretrained('pretrained_model/pretrain_model_up', config = config_up)
        self.lower_model = GPT2LMHeadModel.from_pretrained('pretrained_model/pretrain_model_down', config = config_down)

        #self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        #self.q_attn = Conv1D(self.embed_dim, self.embed_dim)

        #self.linear = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def save_pretrained(self, ckpt_path, epoch):
        upper_model_name = 'nle_upper_model_{}'.format(str(epoch))
        lower_model_name = 'nle_lower_model_{}'.format(str(epoch))
        if epoch % 5 == 0:
            self.upper_model.save_pretrained(ckpt_path + upper_model_name, save_function=accelerator.save)
            self.lower_model.save_pretrained(ckpt_path + lower_model_name, save_function=accelerator.save)

    def forward(self, img_embeddings, input_ques_ans_ids, input_ques_exp_ids, labels4ques_ans, labels4ques_exp, ques_ans_segment_ids, ques_exp_segment_ids):


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
                        test_or_train=None)

        hidden_states = upper_outputs.hidden_states
        #img_embeddings_lower = torch.cat((hidden_states, img_embeddings), dim=1)

        lower_outputs = self.lower_model(
                        up_hidden_states=hidden_states,
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
                        test_or_train=None)

        loss = upper_outputs.loss + lower_outputs.loss



        #kl_loss = nn.KLDivLoss(reduction="batchmean")
        #for upper_weights, lower_weights in zip(self.upper_model.parameters(), self.lower_model.parameters()):
        #    input_dis = F.log_softmax(lower_weights, dim=-1)
        #    target_dis = F.softmax(upper_weights, dim=-1)
        #    loss += kl_loss(input_dis, target_dis)

        return loss
        
        


def sample_sequences(model, tokenizer, loader):
    
    model.eval()
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    max_len = 20
    
    for i,batch in enumerate(loader):
        
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

                hidden_states = outputs.hidden_states
                
                ans_logits = outputs.logits 
                ans_logits = ans_logits[0, -1, :] / temperature
                ans_logits = top_filtering(ans_logits, top_k=top_k, top_p=top_p)
                ans_probs = F.softmax(ans_logits, dim=-1)
                ans_prev = torch.topk(ans_probs, 1)[1] if no_sample else torch.multinomial(ans_probs, 1)
                
                if ans_prev.item() in special_tokens_ids:
                    break
                
                new_ans_segment = special_tokens_ids[-2]
                # take care of when to start the <explanation> token
                # if not always_exp:
                    
                #     if prev.item() != because_token:
                #         new_segment = special_tokens_ids[-2]   # answer segment
                #     else:
                #         new_segment = special_tokens_ids[-1]   # explanation segment
                #         always_exp = True
                # else:
                #     new_segment = special_tokens_ids[-1]   # explanation segment
                    
                new_ans_segment = torch.LongTensor([new_ans_segment]).to(device)
                current_output_ans.append(ans_prev.item())
                input_ques_ans_ids = torch.cat((input_ques_ans_ids, ans_prev.unsqueeze(0)), dim = 1)
                ques_ans_segment_ids = torch.cat((ques_ans_segment_ids, new_ans_segment.unsqueeze(0)), dim = 1)

            for step in range(max_len + 1):
                if step == max_len:
                    break
                
                outputs = model.lower_model(
                                up_hidden_states=hidden_states,
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
                                test_or_train='test')
                
                exp_logits = outputs.logits 
                exp_logits = exp_logits[0, -1, :] / temperature
                exp_logits = top_filtering(exp_logits, top_k=top_k, top_p=top_p)
                exp_probs = F.softmax(exp_logits, dim=-1)
                exp_prev = torch.topk(exp_probs, 1)[1] if no_sample else torch.multinomial(exp_probs, 1)
                
                if exp_prev.item() in special_tokens_ids:
                    break
                
                new_exp_segment = special_tokens_ids[-1]

                new_exp_segment = torch.LongTensor([new_exp_segment]).to(device)
                current_output_exp.append(exp_prev.item())
                input_ques_exp_ids = torch.cat((input_ques_exp_ids, exp_prev.unsqueeze(0)), dim = 1)
                ques_exp_segment_ids = torch.cat((ques_exp_segment_ids, new_exp_segment.unsqueeze(0)), dim = 1)
                
                
        decoded_ans = tokenizer.decode(current_output_ans, skip_special_tokens=True).lstrip()
        decoded_exp = tokenizer.decode(current_output_exp, skip_special_tokens=True).lstrip()

        results_full.append({"image_id": img_id.item(), "caption": decoded_ans + ' because ' + decoded_exp})
        
        results_exp.append({"image_id": img_id.item(), "caption": decoded_exp})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')
            
    return results_full, results_exp

def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


accelerator = Accelerator()
device = accelerator.device

finetune_pretrained = False   # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
ckpt_path = 'ckpts/'
caption_save_path = 'cococaption/results/' 
annFileExp = '/home/mj/nlxgpt/cococaption/annotations/vqaX_test_annot_exp.json'
annFileFull = '/home/mj/nlxgpt/cococaption/annotations/vqaX_test_annot_full.json'
nle_data_train_path = '/home/mj/nlxgpt/nle_data/VQA-X/vqaX_train.json'
nle_data_test_path = '/home/mj/nlxgpt/nle_data/VQA-X/vqaX_test.json'
nle_data_val_path = '/home/mj/nlxgpt/nle_data/VQA-X/vqaX_val.json'
max_seq_len = 40
max_exp_seq_len = 40
load_from_epoch = None
no_sample = True   
top_k =  0
top_p =  0.9
batch_size = 32   # per GPU 
num_train_epochs = 200
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
        
        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>', 'additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
        
        #assert len(tokenizer) == orig_num_tokens + num_new_tokens
        
        model = NLXmodel()
        #model = GPT2LMHeadModel.from_pretrained('/home/lizhengyi/2022/task_basecalling/ex_try/bonito_copy/distillgpt2', config = config)
        model.upper_model.resize_token_embeddings(len(tokenizer))
        model.lower_model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)
        
print("Model Setup Ready...")


img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = VQAXTrainDataset(path = nle_data_train_path, 
                                 transform = img_transform, 
                                 tokenizer = tokenizer, 
                                 max_seq_len = max_seq_len,
                                 max_exp_seq_len = max_exp_seq_len
                                 )

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)

# val_dataset = VQAXEvalDataset(path = nle_data_val_path,      
#                               transform = img_transform, 
#                               tokenizer = tokenizer, 
#                               max_seq_len = max_seq_len)


# val_loader = torch.utils.data.DataLoader(val_dataset,
#                                          batch_size = 1, 
#                                          shuffle=False, 
#                                          pin_memory=True)

test_dataset = VQAXEvalDataset(path = nle_data_test_path,      
                               transform = img_transform, 
                               tokenizer = tokenizer, 
                               max_seq_len = max_seq_len)


test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = 1, 
                                          shuffle=False, 
                                          pin_memory=True)


model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = 0   # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)


for epoch in range(start_epoch, num_train_epochs):

    model.train()
    accum_loss = 0

    for step, batch in enumerate(train_loader):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        # img, _, input_ids, labels, segment_ids = batch
        img, _, input_ques_ans_ids, input_ques_exp_ids, labels4ques_ans, labels4ques_exp, ques_ans_segment_ids, ques_exp_segment_ids = batch
        
        img_embeddings = image_encoder(img)
        
        # outputs = model(input_ids=input_ids, 
        #                 past_key_values=None, 
        #                 attention_mask=None, 
        #                 token_type_ids=segment_ids, 
        #                 position_ids=None, 
        #                 encoder_hidden_states=img_embeddings, 
        #                 encoder_attention_mask=None, 
        #                 labels=labels, 
        #                 use_cache=False, 
        #                 return_dict=True)
        loss = model(img_embeddings = img_embeddings,
                        input_ques_ans_ids = input_ques_ans_ids, 
                        input_ques_exp_ids = input_ques_exp_ids, 
                        labels4ques_ans = labels4ques_ans, 
                        labels4ques_exp = labels4ques_exp, 
                        ques_ans_segment_ids = ques_ans_segment_ids,
                        ques_exp_segment_ids = ques_exp_segment_ids)
        
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        accum_loss += loss.item()
        
        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accelerator.print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch, num_train_epochs, step, len(train_loader), accum_loss), end='          ')
            accum_loss = 0


    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)

    if accelerator.is_main_process:     
        
        results_full, results_exp = sample_sequences(unwrapped_model, tokenizer, test_loader)
        
        resFileExp = caption_save_path + 'captions_exp_' + str(epoch) + '.json'
        unf_resFileExp = caption_save_path + 'unf_captions_exp_' + str(epoch) + '.json'
        unf_resFileFull = caption_save_path + 'unf_captions_full_' + str(epoch) + '.json'
        save_scores_pathExp = caption_save_path + 'scores_exp_' + str(epoch) + '.json'
        
        with open(unf_resFileExp, 'w') as w:
            json.dump(results_exp, w)
            
        with open(unf_resFileFull, 'w') as w:
            json.dump(results_full, w)
         
        # unfiltered results
        # get_scores(annFileExp, unf_resFileExp, save_scores_pathExp)

        # filtered results
        filter_and_get_scores(resFileExp, save_scores_pathExp, results_full, results_exp)
    
