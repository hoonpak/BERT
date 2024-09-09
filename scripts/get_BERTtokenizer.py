from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece as WordPieceModel
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.processors import TemplateProcessing

tokenizer = Tokenizer(WordPieceModel(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
tokenizer.pre_tokenizer = BertPreTokenizer()

file_path = "../dataset/vocab.txt"
with open(file_path, "r") as file:
    lines = file.readlines()
    special_words = [token.strip() for token in lines]

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
trainer = WordPieceTrainer(vocab_size=30522, special_tokens=special_tokens, show_progress=True)
tokenizer.train(["../dataset/empty.txt"], trainer=trainer)

tokenizer.add_tokens(special_words)

# cls_token_id = tokenizer.token_to_id("[CLS]")
# sep_token_id = tokenizer.token_to_id("[SEP]")

# tokenizer.post_processor = TemplateProcessing(
#     single=f"[CLS]:0 $A:0",
#     pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
#     special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
# )

tokenizer.decoder = WordPieceDecoder()

tokenizer.save("../dataset/BERT_Tokenizer.json")

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
print(tokenizer.decode(output.ids))