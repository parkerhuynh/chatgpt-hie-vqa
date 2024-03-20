def create_hie_ann_vocal(self):
        path_files = self.args.stat_ques_list
        stat_ques_list = []
        for path_file in path_files:
            with open(path_file, 'r') as file:
                single_questions = json.load(file)
            stat_ques_list += single_questions
        question_list = {}
        for ques in stat_ques_list:
            question_list[ques["question_id"]] = ques["question"]

        examples = []
        path_files = self.args.stat_ann_list
        
        for path_file in path_files:
            with open(path_file, 'r') as file:
                single_anns = json.load(file)
            examples += single_anns
        
            
        ans2tok, tok2ans = {}, {}
        
        ans_freq_dict = {}
        
        for ans in examples:
            ans_proc = prep_ans(ans['multiple_choice_answer'])
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
            unique_answers = {answer['answer'] for answer in example['answers']}
            for answer_str in unique_answers:
                ans_proc = prep_ans(answer_str)
                if ans_proc in ans2tok:
                    ans_id = ans2tok[ans_proc]
                    total_answer_counts[ans_id] += 1  # Total appearances of each answer
                    
                    question_str = question_list[example["question_id"]]
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
 
        return ans2tok, tok2ans, answer_percentages
    


class CustomLayer(nn.Module):
    def __init__(self, question_type_map_dict):
        super(CustomLayer, self).__init__()
        
        unique_question_types = set()
        for ans_id, q_types in question_type_map_dict.items():
            unique_question_types.update(q_types.keys())

        # Convert to list if necessary and sort (optional but helps in consistent indexing)
        question_type_ids = sorted(list(unique_question_types))
        answer_ids = sorted(list(question_type_map_dict.keys()))
        
        m = len(question_type_ids)  # Number of question types
        n = len(answer_ids)
        QuestionTypeMatrix = torch.zeros((m, n), dtype=torch.float32)
        
        question_type_to_index = {q_id: idx for idx, q_id in enumerate(question_type_ids)}
        answer_to_index = {a_id: idx for idx, a_id in enumerate(answer_ids)}

        # Populate the matrix
        for ans_id, q_type_percents in question_type_map_dict.items():
            ans_index = answer_to_index[ans_id]
            for q_type_id, percent in q_type_percents.items():
                if q_type_id in question_type_to_index:  # Ensure the question type is in our matrix
                    q_type_index = question_type_to_index[q_type_id]
                    QuestionTypeMatrix[q_type_index, ans_index] = percent
        
        self.register_buffer('QuestionTypeMatrix', QuestionTypeMatrix)
        
    def forward(self, question_type_output, vqa_output):
        question_type_output = F.softmax(question_type_output, dim=-1)
        current_device = question_type_output.device
        bz, _ = question_type_output.size()
        question_type_matrix = self.QuestionTypeMatrix.to(current_device).unsqueeze(0).expand(bz, -1, -1)
        
        question_type_output = question_type_output.unsqueeze(1)
        question_type_output = torch.bmm(question_type_output, question_type_matrix).squeeze(1)
        output = question_type_output*vqa_output
        return output