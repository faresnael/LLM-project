import re
class simpleTokenizerv1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode (self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text
    
with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text  = f.read()
print("Total number of characters: ", len(raw_text))
#raw_text = raw_text[:30]
# print(item.split())

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
# for i,j in enumerate(all_words):
#     print("i is: ", i)
#     print("j is: ", j)
vocab = {token:integer for integer,token in enumerate(all_words)}
tokenizer = simpleTokenizerv1(vocab)
text = """" It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""

# ids = tokenizer.encode(text)
# print(ids)
# print (tokenizer.decode(ids))




