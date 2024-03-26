import re
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
    question_type = question_type.replace("|", " ")
    question_type = re.sub('\s+', ' ', question_type).strip()
    # # 
    for word in ["clothing", "object", "material", "shape", "food", "transportation", "pattern", "letter", "drink"]:
        if word in question_type:
            return "object"
    for word in ["age", "animal", "body parts", "gender", "body part", "person", "emotion", "relationship"]:
        if word in question_type:
            return "human"
    for word in ["weather", "season", "time", "unknown", "any", "not", "event", "geometry", "music", "description", 
                 "flavor", "sound", "taste", "other", "comparison", "team", "duration", "sport this response is incorrect please try again", "fitting",
                 "orientation", "facial feature","please provide the question type for the question what does it say after water lane?"]:        
        if word in question_type:
            return "other"
    for word in ["activity", "action"]:        
        if word in question_type:
            return "action"
    for word in ["direction"]:        
        if word in question_type:
            return "location"
    for word in ["yes/no", "binary", "you missed providing an answer for this question",
                 "it seems like there was an error in the provided question please rephrase the question so that it can be categorized accurately",
                 "Is there traffic on the table?", "no"
                 ]:        
        if word in question_type:
            return "yes/no"
    for word in ["number", "you didnt provide a clear question so i am unable to categorize it into one of the question types provided"]:        
        if word in question_type:
            return "number"
    for word in ["sports"]:   
        if word in question_type:
            return "sport"
    for word in ["color"]:   
        if word in question_type:
            return "color"
    for word in ["location", "hand dominance"]:   
        if word in question_type:
            return "location"
    # question_type = question_type.split(",")
    # question_type =  [item.strip() for item in question_type][0]
    return question_type