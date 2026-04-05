import torch
from pathlib import Path
from transformers import GPT2TokenizerFast

class Tokenizer:
    def __init__(self, 
                 config: str | None = None,
                #  vocab_file: str | None = None,
                #  merges_file: str | None = None,
        ) -> None:
        
        """
        A wrapper class for GPT2 tokenizer that handles encoding and decoding of text.
        This class initializes a GPT2TokenizerFast instance with custom or default vocabulary
        and merges files. It automatically sets a padding token if one is not already defined.
        Attributes:
            tokenizer (GPT2TokenizerFast): The underlying GPT2 tokenizer instance.

        Initialize the Tokenizer with vocabulary and merges files.
        If vocab_file or merges_file are not provided, default files from the package
        directory are used. Sets the pad_token to eos_token if not already defined.
        Args:
            vocab_file (str | None): Path to the vocabulary file. Defaults to None,
                which uses the package's default vocab.json.
            merges_file (str | None): Path to the merges file. Defaults to None,
                which uses the package's default merges.txt.
        Returns:
            None
        """
        # module_dir = Path(__file__).resolve().parent
        module_dir = Path(config['model_dir'])
        vocab = module_dir / "vocab.json"
        merges = module_dir / "merges.txt"
        self.tokenizer = GPT2TokenizerFast(vocab_file=vocab, merges_file=merges, clean_up_tokenization_spaces=True)
        # Set padding side to left for GPT2, which is a causal language model. 
        # This ensures that padding tokens are added to the left of the input sequence, 
        # allowing the model to attend to the most recent tokens on the right.
        self.tokenizer.padding_side = "left"
        
        # GPT2 tokenizer does not have a padding token by default, so we set it to the end-of-sequence token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode( self, 
                text: str | list[str],
                return_tensor: bool = True,
                padding: bool = True,
                max_length: int | None = None,
                truncation: bool = True
        ) -> torch.Tensor | list[int] | list[list[int]]:
        
        '''
        A unified method to encode either a single text or a batch of texts into token IDs.
        Args:
            text (str | list[str]): The input text or list of texts to encode.
            return_tensor (bool): Whether to return the encoded tokens as a PyTorch tensor. Defaults to True.
            padding (bool): Whether to pad the encoded token lists to the same length. Defaults to True.
            max_length (int | None): The maximum length of the encoded token lists
            truncation (bool): Whether to truncate the encoded token lists to the max_length. Defaults to True.
        
        Returns:
            torch.Tensor | list[int] | list[list[int]]: The encoded token IDs as a PyTorch tensor, 
                                                        a list of integers for a single text, or a 
                                                        list of lists of integers for a batch of texts, 
                                                        depending on the input and return_tensor flag.
            
        '''
        
        
        if isinstance(text, str):
            return self.single_sentence_encode(text, return_tensor, max_length, truncation)
        elif isinstance(text, list):
            return self.batch_encode(text, return_tensor, padding, max_length, truncation) 
        else:
            raise TypeError('Input must be a string or a list of strings.')
        
    
    def decode( self,
                tokens: list[int] | list[list[int]] | torch.Tensor
        ) -> str | list[str]:
        
        '''
        A unified method to decode either a single list of token IDs or a batch of token ID lists back into text.
        Args:
            tokens: 1D token ids (single) or 2D token ids (batch), as list or tensor.
        Returns:
            str for single sequence, list[str] for batch.
        '''
        
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim == 1:
                return self.single_sentence_decode(tokens.tolist())
            elif tokens.ndim == 2:
                return self.batch_decode(tokens)
            else:
                raise ValueError("Tensor input to decode must be 1D or 2D.")
            
        if isinstance(tokens, list):
            if len(tokens) == 0:
                return ""
            elif isinstance(tokens[0], int):
                return self.single_sentence_decode(tokens)
            elif isinstance(tokens[0], list):
                return self.batch_decode(tokens)
            else:
                raise TypeError("Input must be a 1D/2D tensor, list[int], or list[list[int]].")
        
    def single_sentence_encode( self, 
                text: str, 
                return_tensor: bool = True,
                max_length: int | None = None,
                truncation: bool = True
        ) -> torch.Tensor | list[int]:
        
        """
        Encode text into a list of token IDs.
        Args:
            text (str): The input text to encode.
            return_tensor (bool): Whether to return the encoded tokens as a PyTorch tensor. Defaults to True.
            max_length (int | None): The maximum length of the encoded token list. If None, no truncation is applied. Defaults to None.
            truncation (bool): Whether to truncate the encoded token list to the max_length. Defaults to True.
        Returns:
            torch.Tensor | list[int]: A list of integer token IDs representing the encoded text, 
                                    or a PyTorch tensor if return_tensor is True.

        """
        
        encoded = self.tokenizer(text, 
                                 truncation=truncation,
                                 max_length=max_length,
                                 return_tensors='pt' if return_tensor else None
                    )["input_ids"]
        
        return encoded, None 
    
    def batch_encode( self, 
                      texts: list[str], 
                      return_tensor: bool = True,
                      padding: bool = True,
                      max_length: int | None = None,
                      truncation: bool = True,
        ) -> torch.Tensor | list[list[int]]:
        
        """
        Encode a batch of texts into lists of token IDs.
        Args:
            texts (list[str]): A list of input texts to encode.
            return_tensor (bool): Whether to return the encoded tokens as a PyTorch tensor. Defaults to True.
            padding (bool): Whether to pad the encoded token lists to the same length. Defaults to True.
            truncation (bool): Whether to truncate the encoded token lists to the max_length. Defaults to True.
            max_length (int | None): The maximum length of the encoded token lists
        Returns:
            torch.Tensor | list[list[int]]: A list of lists of integer token IDs 
                representing the encoded texts, or a PyTorch tensor if return_tensor is True.
            torch.Tensor: The attention mask indicating which tokens are padding (0) and 
                which are not (1), if return_tensor is True. Otherwise, None.
        """
        result = self.tokenizer(texts, 
                                padding=padding,
                                max_length=max_length,
                                truncation=truncation,
                                return_tensors='pt' if return_tensor else None,
                                )

        return result["input_ids"], result["attention_mask"] if return_tensor else None
        
    def single_sentence_decode( self, 
                       tokens: list[int],
        ) -> str:
        
        """
        Decode a list of token IDs back into text.
        Args:
            tokens (list[int]): A list of integer token IDs to decode.
        Returns:
            str: The decoded text with special tokens removed.
        """

        decoded = self.tokenizer.decode(tokens, 
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False
                                    )

        return decoded

    def batch_decode( self,
                        batch_tokens: list[list[int]] | torch.Tensor,
        ) -> list[str] | str:
        
        """
        Decode a batch of token ID lists back into texts.
        Args:
            batch_tokens (list[list[int]] | torch.Tensor): A list of lists of integer token IDs or a PyTorch tensor to decode.
        Returns:
            list[str] | str: A list of decoded texts with special tokens removed, or a single decoded text if input is a tensor.
        """
        
        decoded = self.tokenizer.batch_decode(batch_tokens, 
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False
                                            )
        return decoded
