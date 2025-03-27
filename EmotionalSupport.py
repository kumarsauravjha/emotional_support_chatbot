#hide
# Imports

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import logging
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import gradio as  gr

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pathlib import Path
import json
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    #AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers import (T5Tokenizer, 
                              T5ForConditionalGeneration, 
                              T5Config)
#from utils.data_parallel import BalancedDataParallel
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)





# Args to allow for easy convertion of python script to notebook
class Args():
    def __init__(self):
        # self.output_dir = './t5_strategy/checkpoint-2130'
        self.output_dir = './t5_strategy_full'
        self.model_type = 't5'
        self.model_name_or_path = 't5_strategy_full'
        self.config_name = 'google/t5-v1_1-small'
        self.tokenizer_name = 'google/t5-v1_1-small'
        self.data_path = "dataset"
        self.train_file_name = "trainWithStrategy_short.tsv"
        self.eval_file_name = "testWithStrategy_short.tsv"
        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = False
        self.do_eval = False
        self.generation = True
        self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 10   #epochs
        self.max_steps = -1
        self.warmup_steps = 120
        self.logging_steps = 100
        self.save_steps = 500
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.strategy = True
        self.turn = False
        self.role = False
        

class InputFeatures_train(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 role_ids, lm_labels, cls_position, cls_label, strategy_ids, input_len=None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.role_ids = role_ids
        self.lm_labels = lm_labels
        self.cls_position = cls_position
        self.cls_label = cls_label
        self.strategy_ids = strategy_ids
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


class InputFeatures_blender(object):
    def __init__(self, encoder_feature, decoder_feature):
        self.conv_id = encoder_feature.conv_id
        self.input_ids = encoder_feature.input_ids
        self.position_ids = encoder_feature.position_ids
        self.token_type_ids = encoder_feature.token_type_ids
        self.role_ids = encoder_feature.role_ids
        self.lm_labels = encoder_feature.lm_labels
        self.cls_position = encoder_feature.cls_position
        self.cls_label = encoder_feature.cls_label
        self.strategy_ids = encoder_feature.strategy_ids
        self.decoder_input_ids = decoder_feature.input_ids
        self.decoder_position_ids = decoder_feature.position_ids
        self.decoder_token_type_ids = decoder_feature.token_type_ids
        self.decoder_role_ids = decoder_feature.role_ids
        self.decoder_lm_labels = decoder_feature.lm_labels
        self.decoder_cls_position = decoder_feature.cls_position
        self.decoder_cls_label = decoder_feature.cls_label
        self.decoder_strategy_ids = decoder_feature.strategy_ids



def _make_feature(id_, sents, rls, ts, eos, pad=True, block_size=512, strategy_labels=None, evaluate=False, str_embd=False, generation=False):   
    # we did't use role label and turn number in modeling as they did't carry significant improvement. However, codes still remain here.
    if len(sents) == 0:
        return InputFeatures_train([], [], [], [], [],
                            [], [] , [], [])
    input_ids = [i for s in sents for i in s+[eos]]
    input_ids = input_ids
    lm_labels = []
    token_type_ids = []
    roles = []
    strategy_ids = []
    for i, s in enumerate(sents):
        token_type_ids += [ts[i]] * (len(s) + 1)
        flag_str = -1
        if str_embd: #use for strategy embed but currently we treat strategy as token
            strategy_ids += [strategy_labels[-1]] * (len(s) + 1)
        else:
            strategy_ids += [8] * (len(s) + 1)
        if i < len(sents) - 1:
            lm_labels += [-100] * (len(s) + 1)
            roles += [rls[i]] * (len(s) + 1)
        else:
            lm_labels += (  s + [eos])
            roles += [rls[i]] * (len(s) + 1)
    i = len(lm_labels) - 1
    if len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)
    while i >= 0:
        if lm_labels[i] != -100:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    token_type_ids = token_type_ids[:i+1]
    roles = roles[:i+1]
    if not str_embd:
        strategy_ids = [8]*len(input_ids) # strategy is not used
    else:
        strategy_ids = strategy_ids[:i+1]
    if len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)

    assert (len(input_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids))
    # cut according to block size
    if len(input_ids) > block_size:
        cut_index = input_ids.index(eos,-512)
        input_ids = input_ids[cut_index: ]
        token_type_ids = token_type_ids[cut_index: ]
        lm_labels = lm_labels[cut_index: ]
        roles = roles[cut_index: ]
        strategy_ids = strategy_ids[cut_index: ]
    # pad to multiples of 8
    if pad:
        while len(input_ids) % 8 != 0:
            input_ids.append(0)
            token_type_ids.append(0)
            lm_labels.append(-100)
            roles.append(0)
            strategy_ids.append(8)
        assert len(input_ids) % 8 == 0
    position_ids = list(range(len(input_ids)))
    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids))
    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()
    elif len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)
    if True:
        # if it is for generation, the last sentence of context is the last sentence
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos)
    else:
        # if not, the last sentence of context is the second last sentence
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos,input_ids[::-1].index(eos)+1)
    if evaluate and strategy_labels[-1]!=8:
        try:
            lm_labels[lm_labels.index(strategy_labels[-1]+50257+4687)] = -100
        except Exception:
            pass

    feature = InputFeatures_train(id_, input_ids, position_ids, token_type_ids, roles,
                            lm_labels, cls_position , strategy_labels[-1], strategy_ids)
    return feature

def _norm_text(text):
    w, r, t, *toks = text.strip().split()
    try:
        f = int(r)
        t = int(t)
        toks = ' '.join(toks[:len(toks)])
    except Exception as e:
        raise e
    return f, t, toks

def _get_inputs_from_text(text, tokenizer, strategy=True, cls = False):
    srcs = text.strip()
    inputs = []
    roles = []
    turns = []
    strategy_labels=[]
    srcs = srcs.split(" EOS")
    for idx, src in enumerate(srcs):
        if src =="":
            continue
        src_role, src_turn, src = _norm_text(src)
        context_id = tokenizer.encode(src)
        if not strategy:
            context_id = [ i  for i in context_id if i< 50257+4687]
        elif cls:
            context_id = tokenizer.cls + [ i  for i in context_id if i< 50257+4687]
        else:
            pass
        if src_role==1:
            try:
                label = "["+src.split("[")[1].split("]")[0]+"]"
            except Exception as e:
                strategy_labels.append(8)
            else:
                strategy_labels.append( tokenizer.encode([label])[0] - 50257-4687)
        else:
            strategy_labels.append(8)
        inputs.append(context_id)
        roles.append(src_role)
        turns.append(src_turn)
    return inputs, roles, turns, strategy_labels

def construct_conv_ESD(idx, row, tokenizer, eos = True, pad=True, cls=False, evaluate=False, strategy=True, generation=False):
    #  process input text
    inputs, roles, turns, strategy_labels = _get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), tokenizer, strategy=strategy)
    # process output (decoder input) text
    d_inputs, d_roles, d_turns, d_strategy_labels = _get_inputs_from_text(row.split("EOS")[-1], tokenizer, strategy=strategy)
    # make feature for input text
    feature = _make_feature(idx, inputs, roles, turns, tokenizer.eos_token_id, pad=pad, strategy_labels=strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
    # make feature for output (decoder input) text
    d_feature = _make_feature(idx, d_inputs, d_roles, d_turns, tokenizer.eos_token_id, pad=pad, strategy_labels=d_strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
    feature = InputFeatures_blender(feature, d_feature) 
    return feature

class T5Dataset(Dataset):
    def __init__(self, tokenizer, args, dataframe, block_size=512, evaluate=False, strategy=True):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = block_size

        for idx, line in enumerate(dataframe):
            line = line.strip()
            if not line:
                continue

            segments = line.split(" EOS")
            if len(segments) < 2:
                continue

            # Split into context and response (last turn)
            context_segments = segments[:-1]
            response_segment = segments[-1]

            # Extract full formatted context: "role turn text EOS ..."
            formatted_context = []
            for seg in context_segments:
                parts = seg.strip().split()
                if len(parts) < 4:
                    continue
                role = parts[1]
                turn = parts[2]
                utt = " ".join(parts[3:])
                formatted_context.append(f"{role} {turn} {utt.strip()} EOS")
            input_text = " ".join(formatted_context)

            # Extract strategy tag + response
            parts = response_segment.strip().split()
            if len(parts) < 4:
                continue
            role = parts[1]
            turn = parts[2]
            response_utt = " ".join(parts[3:]).strip()

            # Strategy tag: only added if supporter response
            if strategy and role == "1" and "[" in response_utt:
                strategy_tag = response_utt.split("]")[0] + "]"
                input_text = f"{strategy_tag} {input_text}"
            else:
                input_text = f"support: {input_text}"

            # Add [Other] to target if missing
            if strategy and role == "1" and "[" not in response_utt:
                response_utt = "[Other] " + response_utt

            self.examples.append((input_text, response_utt))
            if idx < 5:  # only print the first 5 for sanity-check
                print("==== EXAMPLE", idx, "====")
                print("Input text:  ", input_text)
                print("Target text: ", response_utt)
                print()


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_text, target_text = self.examples[i]

        source = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels
        }



# class T5Dataset(Dataset):
#     def __init__(self, tokenizer, args, dataframe, block_size=512, evaluate=False, strategy=True):
#         self.examples = []
#         self.tokenizer = tokenizer
#         self.max_length = block_size

#         for idx, line in enumerate(dataframe):
#             line = line.strip()
#             if not line:
#                 continue

#             # Split on " EOS " rather than "\t"
#             segments = line.split(" EOS")
#             # We need at least 2 segments to form input and target
#             if len(segments) < 2:
#                 print(f"⚠️ Skipping malformed line {idx}: {line}")
#                 continue

#             # everything except the last segment is the input text
#             input_text = " EOS".join(segments[:-1]).strip()
#             # last segment is the target text
#             target_text = segments[-1].strip()

#             # Optional: add a strategy label up front if the target has e.g. [Other]
#             if strategy and "[" in target_text:
#                 strategy_tag = target_text.split("]")[0] + "]"  # e.g. [Other]
#                 input_text = f"{strategy_tag} {input_text}"
#             else:
#                 input_text = f"support: {input_text}"

#             # If no bracketed strategy label is in the target, you can add a default:
#             if strategy and "[" not in target_text:
#                 target_text = "[Other] " + target_text

#             # Now store as a pair
#             self.examples.append((input_text, target_text))

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         input_text, target_text = self.examples[i]

#         source = self.tokenizer(
#             input_text,
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )

#         target = self.tokenizer(
#             target_text,
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )

#         labels = target["input_ids"].squeeze()
#         labels[labels == self.tokenizer.pad_token_id] = -100

#         return {
#             "input_ids": source["input_ids"].squeeze(),
#             "attention_mask": source["attention_mask"].squeeze(),
#             "labels": labels
#         }

def load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False, strategy=True):
    df = df_val if evaluate else df_trn
    return T5Dataset(tokenizer, args, df, block_size=args.block_size, evaluate=evaluate, strategy=strategy)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

# # Training of model
# def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
#     """ Train the model """
#     if args.local_rank in [-1, 0]:
#         tb_writer = SummaryWriter() #logging metrics like loss, learning rate to TensorBoard.

#     args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
#     def collate(examples: List[torch.Tensor]):
#         if tokenizer._pad_token is None:
#             return pad_sequence(examples, batch_first=True)
#         return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
#     train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
#     # train_dataloader = DataLoader(
#     #     train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=ESDDataset.collate, drop_last = True
#     # )
#     train_dataloader = DataLoader(
#     train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)
#     if args.max_steps > 0:
#         t_total = args.max_steps
#         args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
#     else:
#         t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

#     model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
#     model.resize_token_embeddings(len(tokenizer))
#     # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": args.weight_decay,
#         },
#         {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
#     ]
#     '''The model is standing by. The strategy tokens are in its vocabulary. The optimizer coach is ready. 
#     But before we begin, let’s take inventory.'''
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f'{total_params:,} total parameters.')
#     total_trainable_params = sum(
#     p.numel() for p in model.parameters() if p.requires_grad)
#     print(f'{total_trainable_params:,} training parameters.')    
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
#     )

#     # Check if saved optimizer or scheduler states exist
#     if False and (
#         args.model_name_or_path
#         and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
#         and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
#     ):
#         # Load in optimizer and scheduler states
#         optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

#     if args.fp16:
#         try:
#             from apex import amp
#         except ImportError:
#             raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#         model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

#     # multi-gpu training (should be after apex fp16 initialization)
#     if args.n_gpu > 1:
#         #model = BalancedDataParallel(2,model, dim=0).to(args.device)
#         model = torch.nn.DataParallel(model)

#     # Distributed training (should be after apex fp16 initialization)
#     if args.local_rank != -1:
#         model = torch.nn.parallel.DistributedDataParallel(
#             model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
#         ).to(args.device)

#     # Train!
#     logger.info("***** Running training *****")
#     logger.info("  Num examples = %d", len(train_dataset))
#     logger.info("  Num Epochs = %d", args.num_train_epochs)
#     logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
#     logger.info(
#         "  Total train batch size (w. parallel, distributed & accumulation) = %d",
#         args.train_batch_size
#         * args.gradient_accumulation_steps
#         * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
#     )
#     logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
#     logger.info("  Total optimization steps = %d", t_total)

#     global_step = 0
#     epochs_trained = 0
#     steps_trained_in_current_epoch = 0
 
#     # Check if continuing training from a checkpoint
#     if False and args.model_name_or_path and os.path.exists(args.model_name_or_path):
#         try:
#             # set global_step to gobal_step of last saved checkpoint from model path
#             checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
#             global_step = int(checkpoint_suffix)
#             epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
#             steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

#             logger.info("  Continuing training from checkpoint, will skip to saved global_step")
#             logger.info("  Continuing training from epoch %d", epochs_trained)
#             logger.info("  Continuing training from global step %d", global_step)
#             logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
#         except ValueError:
#             logger.info("  Starting fine-tuning.")

#     tr_loss, logging_loss, tr_ppl, logging_ppl = 0.0, 0.0, 0.0, 0.0
#     tr_loss1, logging_loss1  = 0.0, 0.0
#     model.zero_grad()
#     #train_iterator = range(epochs_trained, int(args.num_train_epochs))
#     train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=True)
#     set_seed(args)  # Added here for reproducibility
#     import numpy as np
#     np.set_printoptions(threshold=np.inf)

    # for _ in train_iterator:
    #     epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
    #     for step, batch in enumerate(epoch_iterator):
    #         #print("step:",step)
    #         # Skip past any already trained steps if resuming training
    #         if steps_trained_in_current_epoch > 0:
    #             steps_trained_in_current_epoch -= 1
    #             continue
    #         input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_turn_ids, decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids = batch
    #         model.train()
    #         if input_ids.shape[1] > 1024: continue
    #         input_ids = input_ids.to(args.device)
    #         turn_ids = turn_ids.to(args.device)
    #         role_ids = role_ids.to(args.device)
    #         decoder_input_ids = decoder_input_ids.to(args.device)
    #         decoder_turn_ids = decoder_turn_ids.to(args.device)
    #         decoder_label_ids = decoder_labels.to(args.device)
    #         decoder_role_ids = decoder_role_ids.to(args.device)
    #         #decoder_cls_labels = decoder_cls_labels.to(args.device)
    #         model.train()
    #         # we did't use role label and turn number in modeling as they did't carry significant improvement. Codes still remain.
    #         if not args.role:
    #             role_ids = None
    #             decoder_role_ids = None
    #         if not args.turn:
    #             turn_ids = None
    #             decoder_role_ids = None
    #         if not args.strategy:
    #             outputs = model(input_ids, attention_mask = input_ids.ne(tokenizer.pad_token_id), decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids, turn_ids=turn_ids, role_ids=role_ids,labels = decoder_label_ids)
    #             ppl = loss = outputs[0]  # model outputs are always tuple in transformers (see doc)                
    #         else:
    #             outputs = model(input_ids, attention_mask = input_ids.ne(tokenizer.pad_token_id), decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids, turn_ids=turn_ids, role_ids=role_ids,labels = decoder_label_ids)
    #             loss = ppl = outputs.loss
    #         if not args.no_cuda and args.n_gpu >= 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #             ppl = ppl.mean()
    #         if args.gradient_accumulation_steps > 1:
    #             loss = loss / args.gradient_accumulation_steps
    #         if args.fp16:
    #             with amp.scale_loss(loss, optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             loss.backward()
    #         tr_loss += loss.item()
    #         tr_ppl += ppl.item() 
    #         if  args.strategy:
    #             tr_loss1 += outputs[1].mean().item()
    #         outputs =0 
    #         loss = 0
    #         ppl = 0 
    #         if (step + 1) % args.gradient_accumulation_steps == 0:
    #             if args.fp16:
    #                 torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
    #             else:
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #             optimizer.step()
    #             scheduler.step()  # Update learning rate schedule
    #             model.zero_grad()
    #             global_step += 1
    #             if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step >t_total*0.0:
    #                 # Log metrics
    #                 if (
    #                     args.local_rank == -1 and args.evaluate_during_training
    #                 ):  # Only evaluate when single GPU otherwise metrics may not average well
    #                     results = evaluate(args, model, tokenizer, args.eval_dataset, "{}-{}".format("checkpoint", global_step))
    #                     for key, value in results.items():
    #                         tb_writer.add_scalar("eval_{}".format(key), value, global_step)
    #                 tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
    #                 tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
    #                 logger.info("lr: %f, step: %d, loss: %f, ppl: %f ", scheduler.get_lr()[0], global_step, (tr_loss - logging_loss) / args.logging_steps,(tr_loss1- logging_loss1)/args.logging_steps)
    #                 logging_loss = tr_loss
    #                 logging_ppl = tr_ppl
    #                 logging_loss1 = tr_loss1
    #             if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 and global_step > t_total*0.0:
    #                 checkpoint_prefix = "checkpoint"
    #                 # Save model checkpoint
    #                 output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
    #                 os.makedirs(output_dir, exist_ok=True)
    #                 model_to_save = (
    #                     model.module if hasattr(model, "module") else model
    #                 )  # Take care of distributed/parallel training
    #                 model_to_save.save_pretrained(output_dir)
    #                 tokenizer.save_pretrained(output_dir)

    #                 torch.save(args, os.path.join(output_dir, "training_args.bin"))
    #                 logger.info("Saving model checkpoint to %s", output_dir)

    #                 _rotate_checkpoints(args, checkpoint_prefix)

    #                 torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    #                 torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    #                 logger.info("Saving optimizer and scheduler states to %s", output_dir)

    #         if args.max_steps > 0 and global_step > args.max_steps:
    #             epoch_iterator.close()
    #             break
    #     if args.max_steps > 0 and global_step > args.max_steps:
    #         train_iterator.close()
    #         break
    # for _ in train_iterator:
    #     epoch_iterator = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)
    #     for step, batch in enumerate(epoch_iterator):
    #         model.train()

    #         input_ids = batch["input_ids"].to(args.device)
    #         attention_mask = batch["attention_mask"].to(args.device)
    #         labels = batch["labels"].to(args.device)

    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs.loss

    #         if args.n_gpu > 1:
    #             loss = loss.mean()
    #         if args.gradient_accumulation_steps > 1:
    #             loss = loss / args.gradient_accumulation_steps

    #         loss.backward()
    #         tr_loss += loss.item()

    #         if (step + 1) % args.gradient_accumulation_steps == 0:
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #             optimizer.step()
    #             scheduler.step()
    #             model.zero_grad()
    #             global_step += 1

    #             if args.logging_steps > 0 and global_step % args.logging_steps == 0:
    #                 tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
    #                 tb_writer.add_scalar("loss", loss.item(), global_step)

    #         if args.max_steps > 0 and global_step > args.max_steps:
    #             break
    #     if args.max_steps > 0 and global_step > args.max_steps:
    #         break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    # return global_step, tr_loss / global_step

def train(args, train_dataset, model, tokenizer):
    """Train the T5 model"""

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

    # Calculate total steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model
    model.resize_token_embeddings(len(tokenizer))

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Device setup
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        ).to(args.device)

    # Training starts
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    for _ in train_iterator:
        for step, batch in enumerate(train_dataloader):
            model.train()
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", loss.item(), global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    # Final report
    avg_loss = tr_loss / global_step
    perplexity = torch.exp(torch.tensor(avg_loss))
    logger.info(f"Training completed. Avg loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")

    return global_step, avg_loss


# Evaluation of some model
# def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    #eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=ESDDataset.collate, drop_last = True
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    #multi-gpu evaluate
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    import numpy as np
    strategy_probs = []
    cls_labels_list = []
    num_samples = []
    for batch in tqdm(eval_dataloader, desc="Evaluating",disable=True):
        input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_turn_ids, decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids = batch
        model.train()
        if input_ids.shape[1] > 1024: continue
        input_ids = input_ids.to(args.device)
        turn_ids = turn_ids.to(args.device)
        role_ids = role_ids.to(args.device)
        decoder_input_ids = decoder_input_ids.to(args.device)
        decoder_turn_ids = decoder_turn_ids.to(args.device)
        decoder_label_ids = decoder_labels.to(args.device)
        decoder_role_ids = decoder_role_ids.to(args.device)
        decoder_cls_labels = decoder_cls_labels.to(args.device)
        if not args.role:
            role_ids = None
            decoder_role_ids = None
        if not args.turn:
            turn_ids = None
            decoder_role_ids = None

        with torch.no_grad():
            if not args.role:
                role_ids = None
            if not args.turn:
                turn_ids = None
            if args.strategy:
                outputs = model(input_ids, decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids, turn_ids=turn_ids, role_ids=role_ids, labels=decoder_label_ids)
                loss = ppl = outputs.loss
            else:
                outputs = model(input_ids, decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids, turn_ids=turn_ids, role_ids=role_ids, labels=decoder_label_ids)
                ppl = loss = outputs[0]
            if args.strategy:    
                cls_labels_list.extend(decoder_cls_labels.cpu().numpy().tolist())         
                strategy_probs.append(torch.nn.functional.softmax(outputs.logits[0, 0, 54945:54945+8], dim=-1).cpu().numpy().tolist())
            lm_loss = outputs[0]
            num_samples.append((decoder_label_ids.cpu().numpy() != -100).astype(np.int).sum())
            eval_loss += lm_loss.sum().item() * (decoder_label_ids.cpu().numpy() != -100).astype(np.int).sum()
        nb_eval_steps += 1
    eval_loss = eval_loss/ sum(num_samples)
    perplexity = torch.exp(torch.tensor(eval_loss))
    np_strategy = np.array(strategy_probs)
    np_cls_labels = np.array(cls_labels_list)
    result = {"perplexity": perplexity}
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("***** Eval results {} *****".format(prefix)+"\n")
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

#collapse




# Main show runner
def evaluate(args, model, tokenizer, eval_dataset, prefix="") -> dict:
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True
    )

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    total_tokens = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()

        # Count only non-padding tokens
        num_tokens = (labels != -100).sum().item()
        eval_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        nb_eval_steps += 1

    eval_loss = eval_loss / total_tokens
    perplexity = torch.exp(torch.tensor(eval_loss))

    logger.info(f"Eval loss = {eval_loss:.4f}, Perplexity = {perplexity:.2f}")

    return {
        "eval_loss": eval_loss,
        "perplexity": perplexity.item()
    }


def main():
    args = Args()
    
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    additional_special_tokens = ["[Questions]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Other]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    config = T5Config.from_pretrained(args.config_name)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    model = T5ForConditionalGeneration.from_pretrained(
    args.model_name_or_path,
    config=args.config_name,
    cache_dir=args.cache_dir )
    
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    logger.info("Training/evaluation parameters %s", args)
    with open(args.data_path+"/"+ args.train_file_name, "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")
   
    with open(args.data_path+"/" + args.eval_file_name, "r", encoding="utf-8") as f:
        df_val = f.read().split("\n")
    
    if args.evaluate_during_training or args.do_eval:
        args.eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True, strategy=args.strategy)
    # Training
    if args.do_train:
        args.train_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False, strategy=args.strategy)
        global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = T5ForConditionalGeneration.from_pretrained(
        args.output_dir,
        config=args.config_name,
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=True
                )

        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)

        tokenizer.add_tokens(additional_special_tokens)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            #model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, args.eval_dataset, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    return results


# def generate():
#     args = Args()
#     additional_special_tokens = ["[Questions]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Other]","[Self-disclosure]","[Affirmation and Reassurance] ","[Providing Suggestions]"]
#     tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
#     tokenizer.add_tokens(additional_special_tokens)
#     tokenizer.add_special_tokens({'cls_token': '[CLS]'})
#     model = T5ForConditionalGeneration.from_pretrained(        
#         args.output_dir,
#         # from_tf=False,
#     )
#     model.resize_token_embeddings(len(tokenizer))
#     if not args.no_cuda:
#         device = torch.device("cuda")
#         args.n_gpu = torch.cuda.device_count()
#         args.device = device
#     else:
#         device = torch.device("cpu")
#         args.device = device
#         args.n_gpu = 0
#     model.to(args.device)
#     chat_history=""
#     # Let's chat for 5 lines
#     start_turn = int(input("which turn do you want to start?(0-26)"))
#     for step in range(20):
#         #chat_history = ""
#         # encode the new user input, add the eos_token and return a tensor in Pytorch
#         usr_inp = input(">> User: ")
#         chat_history = chat_history + "1.0 0 "+ str(start_turn)+" "+ usr_inp+ " EOS "
#         if step > 2: #max num of turn > 5
#             chat_history = "EOS".join(chat_history.split("EOS")[2:]).strip(" ")
#         start_turn += 1
#         f = construct_conv_ESD(step, chat_history, tokenizer, eos = True, pad=False, cls=False, strategy=True , generation=True)
#         paras = {}
#         input_ids = torch.tensor([f.input_ids], dtype=torch.long).to(args.device)
#         paras["attention_mask"] =  input_ids.ne(tokenizer.pad_token_id)
#         chat_history_ids = model.generate(
#             input_ids=input_ids,
#             max_length=1024,min_length=5,
#             pad_token_id= tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,num_beams=1,
#             top_p=0.9, top_k = 30, temperature= 0.7,do_sample=True, **paras,repetition_penalty=1.03)
#         # pretty print last ouput tokens from bot
#         print("EmotionalSupportGPT: {}".format(tokenizer.decode(chat_history_ids[:,:][0], skip_special_tokens=True)))
#         bot_oup = tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True)
#         chat_history = chat_history + "1.0 1 "+ str(start_turn)+" "+bot_oup+ " EOS "
#         start_turn += 1

# def generate():
#     args = Args()
#     additional_special_tokens = [
#         "[Questions]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]",
#         "[Other]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"
#     ]
#     tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
#     tokenizer.add_tokens(additional_special_tokens)
#     tokenizer.add_special_tokens({'cls_token': '[CLS]'})

#     model = T5ForConditionalGeneration.from_pretrained(
#         args.model_name_or_path,
#         ignore_mismatched_sizes=True
#     )
#     model.resize_token_embeddings(len(tokenizer))

#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#     model.to(device)
#     model.eval()

#     print(" Emotional Support ChatBot (T5) is ready! Type 'exit' to quit.")
#     for step in range(20):
#         usr_inp = input(">> User: ")
#         if usr_inp.lower() == "exit":
#             break

#         # input_text = f"support: {usr_inp.strip()}"
#         input_text = f"{usr_inp.strip()}"
#         encodings = tokenizer(
#             input_text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=args.block_size
#         ).to(device)

#         output_ids = model.generate(
#             input_ids=encodings["input_ids"],
#             attention_mask=encodings["attention_mask"],
#             max_length=128,
#             num_beams=4,
#             early_stopping=True,
#             temperature=0.7,
#             top_k=30,
#             top_p=0.9,
#             repetition_penalty=1.03,
#             do_sample=True
#         )

#         decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         print(f"EmotionalSupportGPT: {decoded_output}")

def generate():
    args = Args()
    additional_special_tokens = [
        "[Questions]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
        "[Other]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]"
    ]
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    model.eval()

    print("🧠 Emotional Support ChatBot is ready! Type 'exit' to quit.\n")

    chat_history = ""
    turn_num = 0
    max_turns_to_keep = 6  # 🔁 Limit context size

    def trim_chat_history(history, max_turns=6):
        turns = history.strip().split("EOS")
        turns = [t.strip() for t in turns if t.strip()]
        return " EOS ".join(turns[-max_turns:]) + " EOS "

    for step in range(20):
        usr_inp = input(">> User: ")
        if usr_inp.lower() == "exit":
            break

        # Add user's input to history
        chat_history += f"1.0 0 {turn_num} {usr_inp.strip()} EOS "
        turn_num += 1

        # Trim history before feeding to model
        trimmed_history = trim_chat_history(chat_history, max_turns=max_turns_to_keep)

        input_text = f"support: {trimmed_history.strip()}"
        encodings = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.block_size
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                max_length=128,
                do_sample=True,
                top_p=0.95,
                top_k=30,
                temperature=0.7,
                repetition_penalty=1.03
            )

        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"EmotionalSupportGPT: {decoded_output}")

        # Prepend default strategy if missing
        if not any(tag in decoded_output for tag in additional_special_tokens):
            decoded_output = "[Other] " + decoded_output

        chat_history += f"1.0 1 {turn_num} {decoded_output.strip()} EOS "
        turn_num += 1



def generate_and_evaluate():
    args = Args()

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        args.device = device
        args.n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0

    additional_special_tokens = [
        "[Questions]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
        "[Other]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]"
    ]

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    model.eval()

    # Load evaluation file
    with open(f"{args.data_path}/{args.eval_file_name}", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    references = []
    predictions = []

    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Generating"):
        try:
            input_text, target_text = line.split("\t")
        except:
            continue  # skip malformed lines

        # Format input like during training
        if args.strategy and "[" in target_text:
            strategy_tag = target_text.split("]")[0] + "]"
            input_text = f"{strategy_tag} {input_text}"
        else:
            input_text = f"support: {input_text}"

        encodings = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.block_size
        ).to(args.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                max_length=128,
                do_sample=True,
                top_p=0.95,
                temperature=0.9,
                repetition_penalty=1.03
            )

        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(prediction.strip())
        references.append(target_text.strip())

    # Save outputs
    with open("generated_predictions.json", "w", encoding="utf-8") as f_out:
        json.dump(predictions, f_out, indent=2, ensure_ascii=False)

    with open("ref_strategy.json", "w", encoding="utf-8") as f_out:
        json.dump(references, f_out, indent=2, ensure_ascii=False)

    print("✅ Generation complete. Saved to `generated_predictions.json` and `ref_strategy.json`.")

def predict_emotional_support(user_input, history=[]):
    args = Args()

    additional_special_tokens = [
        "[Questions]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
        "[Other]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]"
    ]

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    model.eval()

    # Build chat history
    chat_history = ""
    turn_num = 0
    for turn in history:
        chat_history += f"1.0 0 {turn_num} {turn[0]} EOS "
        turn_num += 1
        chat_history += f"1.0 1 {turn_num} {turn[1]} EOS "
        turn_num += 1

    chat_history += f"1.0 0 {turn_num} {user_input.strip()} EOS "

    input_text = f"support: {chat_history.strip()}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=args.block_size).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.03
        )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    return reply

# def generate_and_evaluate():
#     args = Args()
#     additional_special_tokens = ["[Questions]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Other]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
#     tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path)
#     tokenizer.add_tokens(additional_special_tokens)
#     tokenizer.add_special_tokens({'cls_token': '[CLS]'})
#     model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.output_dir,
#         from_tf=False)
#     model.resize_token_embeddings(len(tokenizer)) 
#     #model.resize_token_embeddings(54944) 
#     # Setup CUDA, GPU & distributed training
#     if  not args.no_cuda:
#         device = torch.device("cuda")
#         args.n_gpu = torch.cuda.device_count()
#         args.device = device
#     else:
#         device = torch.device("cpu")
#         args.device = device
#         args.n_gpu = 0

#     chat_texts = []
#     with open(args.data_path+"/"+args.eval_file_name,"r") as f:
#         for line in f.readlines():
#             chat_texts.append(line)

#     gts = []
#     refs = []
#     model.to(args.device)
#     # Let's chat for 5 lines
#     for idx, c_text in tqdm(enumerate(chat_texts), desc="Testing"):
#         if "EOS" not in c_text:
#             continue
#         gts.append(c_text.split("EOS")[-1].strip())
#         gts[idx] = " ".join(gts[idx].split(" ")[3:])
#         next_strategy_id = max(tokenizer.encode(gts[idx]))
#         chat_history = c_text
#         f = construct_conv_ESD(idx, chat_history, tokenizer, eos = True, pad=False, cls=False, strategy=False,generation=True)
#         paras = {}
#         input_ids = torch.tensor([f.input_ids], dtype=torch.long).to(args.device)
#         paras["attention_mask"] =  input_ids.ne(tokenizer.pad_token_id)
#         chat_history_ids = model.generate(
#             input_ids,
#             **paras, max_length=1024,min_length=5,num_beams=1,
#             pad_token_id=0,use_cache=True,
#             eos_token_id=tokenizer.eos_token_id, temperature=0.7,
#             top_p=0.9, top_k = 30, do_sample=True, repetition_penalty=1.03).cpu()
#         print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True)))
#         refs.append(tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True))
#     write_path = "./generated_data/generated_pretrained_single_p9t7k3rp3.json"
#     print("write result to:", write_path)
#     with open(wirte_path, "w",encoding="utf-8") as f:
#         json.dump(refs,f,indent=2,ensure_ascii=False)
#     with open("./generated_data/ref_strategy.json","w",encoding="utf-8") as f:
#         json.dump(gts,f,indent=2,ensure_ascii=False)


if __name__ == "__main__":
    # args = Args()
    # if args.generate_and_eval:
    #     generate_and_evaluate()
    # elif args.generation:
    #     generate()
    # else:
    #     main()
    pass

    # Gradio Interface
     # Gradio Interface
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Your Message")
        clear = gr.Button("Clear Chat")

        def respond(message, chat_history):
            if chat_history is None:
                chat_history = []
            reply = predict_emotional_support(
                message,
                [(m["content"], r["content"]) for m, r in zip(chat_history[::2], chat_history[1::2])]
            )
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": reply})
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot)

    demo.launch()


