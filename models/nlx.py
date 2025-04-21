class NLXmodel(nn.Module):

    def __init__(self):
        super(NLXmodel, self).__init__()
        config = AutoConfig.from_pretrained('/home/intership/nlxgpt_y/distilgpt2')

        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)
        config.img_size = img_size
        config.max_seq_len = max_seq_len
        config.add_cross_attention = True

        self.upper_model = GPT2LMHeadModel.from_pretrained('/home/intership/nlxgpt_y/distilgpt2', config=config)
        self.lower_model = GPT2LMHeadModel.from_pretrained('/home/intership/nlxgpt_y/distilgpt2', config=config)

    def save_pretrained(self, ckpt_path, epoch):
        upper_model_name = 'nle_upper_model_{}'.format(str(epoch))
        lower_model_name = 'nle_lower_model_{}'.format(str(epoch))
        self.upper_model.save_pretrained(ckpt_path + upper_model_name, save_function=accelerator.save)
        self.lower_model.save_pretrained(ckpt_path + lower_model_name, save_function=accelerator.save)

    def forward(self, img_embeddings, input_ques_ans_ids, input_ques_exp_ids, labels4ques_ans, labels4ques_exp,
                ques_ans_segment_ids, ques_exp_segment_ids):
        upper_outputs = self.upper_model(input_ids=input_ques_ans_ids,
                                         past_key_values=None,
                                         attention_mask=None,
                                         token_type_ids=ques_ans_segment_ids,
                                         position_ids=None,
                                         encoder_hidden_states=img_embeddings,
                                         encoder_attention_mask=None,
                                         labels=labels4ques_ans,
                                         use_cache=False,
                                         return_dict=True)

        lower_outputs = self.lower_model(input_ids=input_ques_exp_ids,
                                         past_key_values=None,
                                         attention_mask=None,
                                         token_type_ids=ques_exp_segment_ids,
                                         position_ids=None,
                                         encoder_hidden_states=img_embeddings,
                                         encoder_attention_mask=None,
                                         labels=labels4ques_exp,
                                         use_cache=False,
                                         return_dict=True)

        loss = upper_outputs.loss + lower_outputs.loss

        kl_loss = nn.KLDivLoss(reduction="batchmean")
        for upper_weights, lower_weights in zip(self.upper_model.parameters(), self.lower_model.parameters()):
            input_dis = F.log_softmax(lower_weights, dim=-1)
            target_dis = F.softmax(upper_weights, dim=-1)
            loss += kl_loss(input_dis, target_dis)

        return loss


img_size = 224
max_seq_len = 40