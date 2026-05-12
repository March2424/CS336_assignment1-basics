import regex as re

def process_encode_text(chunk:str, special_tokens: list[str], keep_special_tokens:bool) -> list[list[bytes]]:

    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    _COMPILED_PAT = re.compile(GPT2_PAT)
    
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if keep_special_tokens and chunk:
        pattern = f"({pattern})"
    documents = re.split(pattern, chunk) if pattern else [chunk]

    res_words_bytes: list[list[bytes]] = []
    for segment in documents:
        if keep_special_tokens and segment in special_tokens:
            token_bytes = [segment.encode("utf-8")]
            res_words_bytes.append(token_bytes)
        else:
            tokens = [match.group[0].encode("utf-8") for match in re.finditer(_COMPILED_PAT,segment)]
            for token in tokens:
                word_tokens = [bytes([t]) for t in token]
                res_words_bytes.append(word_tokens)
    return res_words_bytes


class Tokenizer:
    def __init__(self,vocab: dict[int,bytes], 
                merges: list[tuple[bytes,bytes]],
                special_tokens: list[str] | None = None) -> None:
        '''
        Initialize the Tokenizer with a vocabulary, BPE merge rules, and optional special tokens.
        '''
        self.vocab = vocab
        self.vocab_reversed = {v:k for k, v in self.vocab.items()}
        self.merges = merges
        self.merges_ranks = {pair: i for i,pair in enumerate(merges)}
        self.special_tokens = special_tokens or []

        
    @classmethod
    def from_files(cls, vocab_filepath: str,
                        merges_filepath: str,
                        special_tokens: list[str] | None = None):
        """
        Construct a Tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath (str): Path to the vocabulary file (from BPE training).
            merges_filepath (str): Path to the merges file (from BPE training).
            special_tokens (list[str] | None): Optional list of special tokens to include.

        Returns:
            Tokenizer: A Tokenizer instance initialized with the given files.
        """
        vocab : dict[int,bytes] = {}
        with open(vocab_filepath,"r",encoding="utf-8") as f:
            for line in f:
                int_str, token_str = line.strip().split("\t")
                vocab[int(int_str)] = eval(token_str).encode("utf-8")

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath,"r",encoding="utf-8") as f:
            for line in f:
                pairs = line.strip().split()
                if len(pairs) == 2:
                    merges.append((eval(pairs[0]).encode("utf-8"), eval(pairs[1]).encode("utf-8")))
        return cls(vocab = vocab,merges = merges, special_tokens = special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input string into a list of token IDs using the BPE algorithm.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        pre_token_list = process_encode_text(text,self.special_tokens,True)
        
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return 
    
    def decode(self, ids: list[int]) -> str:
        return 
    

        
