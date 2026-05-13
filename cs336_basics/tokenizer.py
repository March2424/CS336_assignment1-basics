import regex as re
import ast


def process_encode_text(chunk:str, special_tokens: list[str], keep_special_tokens:bool) -> list[list[bytes]]:
    if keep_special_tokens and special_tokens:
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        pattern = f"({pattern})"
        documents = re.split(pattern, chunk)
    else:
        # 如果不保留特殊字符，则不进行切分，全部当做普通文本处理
        documents = [chunk]

    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    res_words_bytes: list[list[bytes]] = []
    for segment in documents:
        if not segment:
            continue
        
        if keep_special_tokens and segment in special_tokens:
            token_bytes = [segment.encode("utf-8")]
            res_words_bytes.append(token_bytes)
        else:
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(GPT2_PAT,segment,flags=re.V1)]
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

        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split("\t")
                vocab[int(id_str)] = eval(token_str).encode("utf-8")

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((eval(parts[0]).encode("utf-8"), eval(parts[1]).encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input string into a list of token IDs using the BPE algorithm.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        tokens_id = []
        # list[list[bytes]]
        pre_token_list = process_encode_text(text,self.special_tokens,True)
        for tokens in pre_token_list:
            if len(tokens) == 1 and tokens[0] in self.vocab_reversed:
                token_id = self.vocab_reversed.get(tokens[0])
                tokens_id.append(token_id)
                continue
            while len(tokens) >= 2:
                best_pair = None
                min_rank = float("inf")
                for i in range(len(tokens)-1):
                    pair = (tokens[i],tokens[i+1])
                    if pair in self.merges_ranks:
                        rank = self.merges_ranks[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair
                if best_pair is None:
                    break
                i = 0
                new_token_pair = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i],tokens[i+1]) == best_pair:
                        new_token_pair.append(best_pair[0]+best_pair[1])
                        i += 2
                    else:
                        new_token_pair.append(tokens[i])
                        i+=1
                tokens = new_token_pair
            for part in tokens:
                token_id = self.vocab_reversed.get(part)
                tokens_id.append(token_id)
        return tokens_id
            
    
    def encode_iterable(self, iterable: list[str]) -> iter:
        for chunk in iterable:
            ids = self.encode(chunk) 
            for token_id in ids:
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        tokens = b"".join(self.vocab.get(token_id,b'\xef\xbf\xbd') for token_id in ids) 
        return tokens.decode(encoding="utf-8", errors = "replace")
    

        
