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
            
        question_type_map = {}
        for ans in examples:
            unique_answers = {answer['answer'] for answer in ans['answers']}
            for answer_str in unique_answers:
                ans_proc = prep_ans(answer_str)
                if ans_proc in list(ans2tok.keys()):
                    ans_id = ans2tok[ans_proc]
                    quetion_str = question_list[ans["question_id"]]
                    question_type_str  = self.question_type_dict[quetion_str]
                    question_type_id = self.question_type_to_idx[question_type_str]
                    
                    if ans_id not in question_type_map:
                        question_type_map[ans_id] = []
                    if question_type_id not in question_type_map[ans_id]:
                        question_type_map[ans_id].append(question_type_id)
        
                
        return ans2tok, tok2ans, question_type_map