import nltk
from nltk.tokenize import word_tokenize

# Download punkt tokenizer if not already downloaded
nltk.download('punkt')

class voc:
    
    PAD_Token = 0  # Define PAD_Token within the class
    
    def __init__(self):
        self.num_words = 1  # 0 is reserved for padding 
        self.num_tags = 0
        self.tags = {}
        self.index2tags = {}
        self.questions = {}
        self.word2index = {}
        self.response = {}
  
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.num_words += 1

    def addTags(self, tag):
        if tag not in self.tags:
            self.tags[tag] = self.num_tags
            self.index2tags[self.num_tags] = tag
            self.num_tags += 1
            
    def addQuestion(self, question, answer):
        self.questions[question] = answer
        words = self.tokenization(question)
        for word in words:
            self.addWord(word)
                 
    def tokenization(self, ques):
        tokens = word_tokenize(ques)
        token_list = []
        for token in tokens:
            token_list.append(token.lower())
        return token_list
    
    def getIndexOfWord(self, word):
        return self.word2index.get(word, self.PAD_Token)
    
    def getQuestionInNum(self, ques):
        words = self.tokenization(ques)
        print(f"Tokenized words: {words}")  # Debugging line
        tmp = [0 for _ in range(self.num_words)]
        for word in words:
            tmp[self.getIndexOfWord(word)] = 1
        print(f"Question in numerical format: {tmp}")  # Debugging line
        return tmp
    
    def getTag(self, tag):
        tmp = [0.0 for _ in range(self.num_tags)]
        tmp[self.tags[tag]] = 1.0
        return tmp
    
    def getVocabSize(self):
        return self.num_words
    
    def getTagSize(self):
        return self.num_tags

    def addResponse(self, tag, responses):
        self.response[tag] = responses
