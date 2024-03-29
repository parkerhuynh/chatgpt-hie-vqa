import json
from torch.utils.data import Dataset, DataLoader
import torch
import os
import pickle
import en_core_web_lg, random, re, json
import numpy as np
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets.randaugment import RandomAugment
import random
from collections import defaultdict
contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}
def LSTM_tokenize(stat_ques_list, args):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }
    
    spacy_tool = None
    pretrained_emb = []
    if args.use_glove:
        spacy_tool = en_core_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if args.use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)
    return token_to_ix, pretrained_emb

def rnn_proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques.lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break
    return ques_ix
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', #'-',
                '>', '<', '@', '`', ',', '?', '!']
def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
            or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText
comma_strip = re.compile("(\d)(\,)(\d)")
manual_map = { 'none': '0',
                'zero': '0',
                'one': '1',
                'two': '2',
                'three': '3',
                'four': '4',
                'five': '5',
                'six': '6',
                'seven': '7',
                'eight': '8',
                'nine': '9',
                'ten': '10'}
articles = ['a', 'an', 'the']

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def prep_ans(answer):
    
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

class VQADataset(Dataset):
    def __init__(self, args, split, rank):
        self.rank = rank
        self.args = args
        self.split = split
        self.image_path = getattr(args, f"{split}_image_path")
        self.saved_image_path = getattr(args, f"{split}_saved_image_path")
        
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        if split == "train":
            # self.transform = transforms.Compose([
            #     transforms.RandomResizedCrop(args.img_size, scale=(0.5, 1.0),
            #                                 interpolation=Image.BICUBIC),
            #     RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
            #                                         'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            #     transforms.ToTensor(),
            #     normalize,
            # ])
            self.transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
            
        for folder_idx in range(20, 35):
            os.makedirs(os.path.join(self.saved_image_path, f"S{str(folder_idx)}"), exist_ok=True)
            
    
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.question_type_to_idx = {
            'yes/no': 0,
            'color': 1,
            'object': 2,
            'number': 3,
            'location': 4,
            'action': 5,
            'other': 6,
            'human': 7,
            'sport': 8
            }
        self.idx_to_question_type = {
            0: 'yes/no',
            1: 'color',
            2: 'object',
            3: 'number',
            4: 'location',
            5: 'action',
            6: 'other',
            7: 'human',
            8: 'sport'
            }
        self.question_type_dict = self.load_question_type()
        
        
        if self.split in  ["val", "test"]:
            self.token_to_ix, self.pretrained_emb = pickle.load(open(self.args.question_dict, 'rb'))
            if self.rank == 0:
                print(f"    - {self.split} : Loaded question vocabulary")
        else:
            if os.path.exists(self.args.question_dict):  
                self.token_to_ix, self.pretrained_emb = pickle.load(open(self.args.question_dict, 'rb'))
                if self.rank == 0:
                    print(f"    - {self.split} : Loaded question vocabulary")
            else:
                self.token_to_ix, self.pretrained_emb = self.prepare_question_vocab()
                pickle.dump([self.token_to_ix, self.pretrained_emb ], open(self.args.question_dict, 'wb'))
                if self.rank == 0:
                    print(f"    - {self.split} : Created question vocabulary")
              
        self.questions = self.load_questions()
        if self.split in  ["val", "test"]:
            self.vqa_ans_to_idx, self.idx_to_vqa_ans, self.question_type_map = json.load(open(self.args.answer_dict, 'r'))
            if self.rank == 0:
                print(f"    - {self.split} : Loaded vqa answer vocabulary")
        else:
            if os.path.exists(self.args.answer_dict):
                self.vqa_ans_to_idx, self.idx_to_vqa_ans, self.question_type_map = json.load(open(self.args.answer_dict, 'r'))
                if self.rank == 0:
                    print(f"    - {self.split} : Loaded vqa answer vocabulary")
            else:
                if "hie" in args.model_name.lower():
                    self.create_hie_ann_vocal = self.create_probability_hie_ann_vocal if args.layer_name == "probability" else self.create_onehot_hie_ann_vocal
                    self.vqa_ans_to_idx, self.idx_to_vqa_ans, self.question_type_map  = self.create_hie_ann_vocal()
                    
                    json.dump([self.vqa_ans_to_idx, self.idx_to_vqa_ans, self.question_type_map], open(self.args.answer_dict, 'w'))
                else:
                    self.vqa_ans_to_idx, self.idx_to_vqa_ans, self.question_type_map = self.create_normal_ann_vocal()
                    json.dump([self.vqa_ans_to_idx, self.idx_to_vqa_ans, self.question_type_map], open(self.args.answer_dict, 'w'))

                if self.rank == 0:
                    print(f"    - {self.split} : Created vqa answer vocabulary")
        
        self.question_type_output_dim = len(self.question_type_to_idx.keys())
           
        
        if self.split in ["train", "val"]:
            self.annotations = self.load_annotations()
            random.shuffle(self.annotations)
        self.token_size = len(self.token_to_ix)
        self.vqa_output_dim = len(self.vqa_ans_to_idx.keys())

    def __len__(self):
        if self.args.debug:
            return 128
        return len(self.annotations) if self.split in ["val", "train"] else len(self.questions)

    def __getitem__(self, idx):
        
        
        if self.split in ["train", "val"]:
            
            
            ann = self.annotations[idx]
            question_id = ann["id"]
            que = self.questions[question_id]
            
            question = que["question"]
            question_type_ids = que["question_type"]
            question_type_str = que["question_type_str"]
            question_onehot = torch.from_numpy(que["onehot_features"])
            encoding = self.bert_tokenizer.encode_plus(
                question,
                add_special_tokens=True,
                max_length=self.args.max_ques_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            ans_iter, answer_str = proc_ans(ann, self.vqa_ans_to_idx)
            image = image_preprocessing(que["img_path"], que["saved_img_path"], self.transform)
            # print("-"*100)
            # print(ans_iter)
            # print(answer_str)
            example = {
                'img_path': que["img_path"],
                'question_id': question_id,
                'question_text': question,
                'bert_input_ids': encoding['input_ids'].flatten(),
                'bert_attention_mask': encoding['attention_mask'].flatten(),
                'onehot_feature': question_onehot,
                'question_type_str': question_type_str,
                'question_type_label': torch.tensor(question_type_ids, dtype=torch.long),
                'vqa_answer_str': answer_str,
                'vqa_answer_label': torch.tensor(ans_iter, dtype=torch.long),
                # 'vqa_answer_label': torch.from_numpy(ans_iter) ,
                "image": image
            }
        else:
            que = self.questions[idx]
            question_id = que["id"]
            question = que["question"]
            question_onehot = torch.from_numpy(que["onehot_features"])
            encoding = self.bert_tokenizer.encode_plus(
                question,
                add_special_tokens=True,
                max_length=self.args.max_ques_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            image = image_preprocessing(que["img_path"], que["saved_img_path"], self.transform)
            example = {
                'img_path': que["img_path"],
                'question_id': question_id,
                'question_text': question,
                'bert_input_ids': encoding['input_ids'].flatten(),
                'bert_attention_mask': encoding['attention_mask'].flatten(),
                'onehot_feature': question_onehot,
                "image": image
            }
            
        return example
    
    def load_questions(self):
        folder_name = {
            "train":"train_images", 
            "val": "val_images",
            "test": "test_images"
        }
        question_path = getattr(self.args, f"{self.split}_question")
        with open(question_path, 'r') as file:
            questions = json.load(file)["questions"]
        
        if self.split in  ["train", "val"]:
            processed_questions = {}
            for question in questions:
                question['img_path'] = os.path.join(self.image_path, question['img_path'])
                question['saved_img_path'] = question['img_path'].replace(folder_name[self.split], "saved_" + folder_name[self.split])
                question['saved_img_path'] = question['saved_img_path'].replace("jpg", "pkl")
                question_type_str = self.question_type_dict[question["question"]]
                question["question_type_str"] = question_type_str
                question_type_idx = self.question_type_to_idx[question_type_str]
                question["question_type"] = question_type_idx
                question["onehot_features"] = rnn_proc_ques(question["question"], self.token_to_ix, 20)
                processed_questions[question["id"]] = question
        else:
            processed_questions = []
            for question in questions:
                question['img_path'] = os.path.join(self.image_path, question['img_path'])
                question['saved_img_path'] = question['img_path'].replace(folder_name[self.split], "saved_" + folder_name[self.split])
                question['saved_img_path'] = question['saved_img_path'].replace("jpg", "pkl")

                question["onehot_features"] = rnn_proc_ques(question["question"], self.token_to_ix, 20)
                processed_questions.append(question) 
        return processed_questions

    def prepare_question_vocab(self):
        """
        Prepare the vocabulary for question processing.
        """
        path_files = self.args.stat_ques_list
        stat_ques_list = []
        for path_file in path_files:
            with open(path_file, 'r') as file:
                single_questions = json.load(file)
            stat_ques_list += single_questions['questions']

        token_to_ix, pretrained_emb = tokenize(stat_ques_list, self.args)
        return token_to_ix, pretrained_emb
    
    def load_annotations(self):
        annotation_path = self.image_path = getattr(self.args, f"{self.split}_annotation")
        with open(annotation_path, 'r') as file:
            annotation_list = json.load(file)["annotations"]
            
        processed_annotations = []
        for ann in annotation_list:
            answer_str = prep_ans(ann["answer"])
            if answer_str in list(self.vqa_ans_to_idx.keys()):
                ans_count = 0
                for judge in ann["judgements"].values():
                    if judge["answer"] == 1:
                        ans_count += 1
                if ans_count >= 2:
                    ann["answer"] = prep_ans(ann["answer"])
                    ann["original_answer"] = ann["answer"]
                    ans_iter = proc_ans(ann, self.vqa_ans_to_idx)
                    ann["ans_iter"] = ans_iter
                    processed_annotations.append(ann)
        
        return processed_annotations
    
    def create_normal_ann_vocal(self):
        examples = []
        path_files = self.args.stat_ann_list
        
        for path_file in path_files:
            with open(path_file, 'r') as file:
                single_anns = json.load(file)
            examples += single_anns['annotations']
            
        processed_examples = []
        for ann in examples:
            ans_count = 0
            for judge in ann["judgements"].values():
                if judge["answer"] == 1:
                    ans_count += 1
            if ans_count >= 2:
                ann["answer"] = prep_ans(ann["answer"])
                ann["original_answer"] = ann["answer"]
                processed_examples.append(ann)
        examples = processed_examples
        ans2tok, tok2ans = {}, {}
        ans_freq_dict = {}
        
        for ans in examples:
            ans_proc = prep_ans(ans['answer'])
            if ans_proc not in ans_freq_dict:
                ans_freq_dict[ans_proc] = 1
            else:
                ans_freq_dict[ans_proc] += 1
        ans_freq_filter = ans_freq_dict.copy()
        most_frequent_words = dict(sorted(ans_freq_dict.items(), key=lambda item: item[1], reverse=True)[:1000])
        ans_freq_filter = most_frequent_words
        for ans in ans_freq_filter:
            tok2ans[tok2ans.__len__()] = ans
            ans2tok[ans] = ans2tok.__len__()

        return ans2tok, tok2ans, {}
    
    def create_onehot_hie_ann_vocal(self):
        path_files = self.args.stat_ques_list
        stat_ques_list = []
        for path_file in path_files:
            with open(path_file, 'r') as file:
                single_questions = json.load(file)
            stat_ques_list += single_questions['questions']
        question_list = {}
        for ques in stat_ques_list:
            question_list[ques["id"]] = ques["question"]

        examples = []
        path_files = self.args.stat_ann_list
        
        for path_file in path_files:
            with open(path_file, 'r') as file:
                single_anns = json.load(file)
            examples += single_anns['annotations']
        
        processed_examples = []
        for ann in examples:
            ans_count = 0
            for judge in ann["judgements"].values():
                if judge["answer"] == 1:
                    ans_count += 1
            if ans_count >= 2:
                ann["answer"] = prep_ans(ann["answer"])
                ann["original_answer"] = ann["answer"]
                processed_examples.append(ann)
        examples = processed_examples
        ans2tok, tok2ans = {}, {}
        
        ans_freq_dict = {}
        
        for ans in examples:
            ans_proc = prep_ans(ans['answer'])
            if ans_proc not in ans_freq_dict:
                ans_freq_dict[ans_proc] = 1
            else:
                ans_freq_dict[ans_proc] += 1
        ans_freq_filter = ans_freq_dict.copy()
        
        most_frequent_words = dict(sorted(ans_freq_dict.items(), key=lambda item: item[1], reverse=True)[:1000])
        ans_freq_filter = most_frequent_words
        
        for ans in ans_freq_filter:
            tok2ans[tok2ans.__len__()] = ans
            ans2tok[ans] = ans2tok.__len__()
            
        question_type_map = {}
        for ans in examples:
            ans_proc = prep_ans(ans['answer'])
            if ans_proc in list(ans2tok.keys()):
                ans_id = ans2tok[ans_proc]
                quetion_str = question_list[ans["id"]]
                question_type_str  = self.question_type_dict[quetion_str]
                question_type_id = self.question_type_to_idx[question_type_str]
                
                if ans_id not in question_type_map:
                    question_type_map[ans_id] = []
                if question_type_id not in question_type_map[ans_id]:
                    question_type_map[ans_id].append(question_type_id)
            
        return ans2tok, tok2ans, question_type_map
    
    
    def create_probability_hie_ann_vocal(self):
        path_files = self.args.stat_ques_list
        stat_ques_list = []
        for path_file in path_files:
            with open(path_file, 'r') as file:
                single_questions = json.load(file)["questions"]
            stat_ques_list += single_questions
        question_list = {}
        for ques in stat_ques_list:
            question_list[ques["id"]] = ques["question"]

        examples = []
        path_files = self.args.stat_ann_list
        
        for path_file in path_files:
            with open(path_file, 'r') as file:
                single_anns = json.load(file)["annotations"]
            examples += single_anns
        ans2tok, tok2ans = {}, {}
        ans_freq_dict = {}
        
        processed_examples = []
        for ann in examples:
            ans_count = 0
            for judge in ann["judgements"].values():
                if judge["answer"] == 1:
                    ans_count += 1
            if ans_count >= 2:
                ann["answer"] = prep_ans(ann["answer"])
                processed_examples.append(ann)
        examples = processed_examples
        
        for ans in examples:
            ans_proc = prep_ans(ans['answer'])
            if ans_proc not in ans_freq_dict:
                ans_freq_dict[ans_proc] = 1
            else:
                ans_freq_dict[ans_proc] += 1
        ans_freq_filter = ans_freq_dict.copy()
        most_frequent_words = dict(sorted(ans_freq_dict.items(), key=lambda item: item[1], reverse=True)[:1000])
        ans_freq_filter = most_frequent_words
        for ans in ans_freq_filter:
            tok2ans[tok2ans.__len__()] = ans
            ans2tok[ans] = ans2tok.__len__()
        
        answer_question_type_counts = defaultdict(lambda: defaultdict(int))
        total_answer_counts = defaultdict(int)
        for example in examples:
            answer_str = example["answer"]
            ans_proc = prep_ans(answer_str)
            if ans_proc in ans2tok:
                ans_id = ans2tok[ans_proc]
                total_answer_counts[ans_id] += 1  # Total appearances of each answer
                
                question_str = question_list[example["id"]]
                question_type_str = self.question_type_dict[question_str]
                question_type_id = self.question_type_to_idx[question_type_str]
                # Count appearances of each answer across different question types
                answer_question_type_counts[ans_id][question_type_id] += 1
        answer_percentages = defaultdict(dict)

        for ans_id, question_type_counts in answer_question_type_counts.items():
            total_appearances = total_answer_counts[ans_id]
            for question_type_id, count in question_type_counts.items():
                percentage = (count / total_appearances)
                answer_percentages[ans_id][question_type_id] = percentage
        print(answer_percentages)
        return ans2tok, tok2ans, answer_percentages
    
    def load_question_type(self):
        question_type_dict = {}
        with open(self.args.question_type, 'r') as file:
            for line in file:
                question_object = json.loads(line)
                question_str = question_object["question"]
                question_type = question_object["question_type"]
                question_type = question_type_processing(question_type)
                question_type_dict[question_str] = question_type
        file.close()
        return question_type_dict
        

def question_type_processing(question_type):
    question_type = question_type.strip()
    question_type = question_type.split(':')[-1]
    question_type = question_type.lower()
    question_type = re.sub(r'\([^)]*\)', '', question_type)
    question_type = question_type.replace("'", "")
    question_type = question_type.replace(".", "")
    question_type = question_type.replace("`", "")
    question_type = question_type.replace(";", ",")
    question_type = question_type.replace('"', "")
    question_type = question_type.replace("question type: ", "")
    question_type = question_type.replace("question: ", "")
    question_type = question_type.replace("’", "")
    question_type = question_type.replace("[", "")
    question_type = question_type.replace("]", "")
    question_type = question_type.replace("-", "")
    question_type = question_type.replace(" or ", ", ")
    question_type = question_type.replace("+", ", ")
    question_type = question_type.replace(" , ", ", ")
    question_type = re.sub('\s+', ' ', question_type).strip()
    # 
    for word in ["clothing", "object", "material", "shape", "food", "transportation", "pattern", "letter", "drink"]:
        if word in question_type:
            return "object"
    for word in ["age", "animal", "body parts", "gender", "body part", "person", "emotion"]:
        if word in question_type:
            return "human"
    for word in ["weather", "season", "time", "unknown", "any", "not", "event", "geometry", "music", "description", "flavor", "sound", "taste"]:        
        if word in question_type:
            return "other"
    for word in ["activity", "action"]:        
        if word in question_type:
            return "action"
    for word in ["direction"]:        
        if word in question_type:
            return "location"
    for word in ["yes/no"]:        
        if word in question_type:
            return "yes/no"
    for word in ["number"]:        
        if word in question_type:
            return "number"
    question_type = question_type.split(",")
    question_type =  [item.strip() for item in question_type][0]
    return question_type

def tokenize(stat_ques_list, args):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if args.use_glove:
        spacy_tool = en_core_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if args.use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)
    return token_to_ix, pretrained_emb


def rnn_proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques.lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break
    return ques_ix

def image_preprocessing(image_path, saved_image_path, transform):
    if os.path.exists(saved_image_path):
        image = pickle.load(open(saved_image_path, 'rb'))
    else:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        pickle.dump(image, open(saved_image_path, 'wb'))
        print(f"saving {saved_image_path}")
    # image = Image.open(image_path).convert('RGB')
    
    return image

def proc_ans(ans, ans_to_ix):
    # ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_proc = prep_ans(ans['answer'])
    
    # ans_score[ans_to_ix[ans_proc]] = 1
    return ans_to_ix[ans_proc], ans_proc

    