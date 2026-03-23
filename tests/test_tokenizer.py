import pytest
from llm_engine.tokenizer.tokenizer import Tokenizer
from llm_engine.config.config_loader import load_asset_paths

config, _ = load_asset_paths()
tokenizer = Tokenizer(config)

def run_tokenizer_case(text: str | list[str], 
                   return_tensor: bool = False, 
                   padding = True, 
                   max_length: int = 512, 
                   truncation = True) -> None:
    
    '''
    Test the tokenizer by encoding and decoding a given text. The decoded text should match the original text.
    
    Args:
        text (str | list[str]): The text to be tokenized and decoded. Can be a single string or a list of strings.
        return_tensor (bool): Whether to return the tokens as a tensor. Default is False.
        padding (bool): Whether to pad the tokens to the maximum length. Default is True.
        max_length (int): The maximum length of the tokenized sequence. Default is 512.
        truncation (bool): Whether to truncate the tokens to the maximum length. Default is True
        
    Raises:
        AssertionError: If the decoded text does not match the original text.
    
    Returns:
        None
    '''
    
    tokens = tokenizer.encode(text, 
                              return_tensor = return_tensor, 
                              padding = padding,
                              max_length = max_length,
                              truncation = truncation
                              )
    
    decoded_text = tokenizer.decode(tokens)
    if return_tensor and not isinstance(text, list): decoded_text = decoded_text[0]
    assert decoded_text == text, (
                                    f"Decoded text does not match original text. "
                                    f"Decoded: {decoded_text}, Original: {text}"
                                )

# @pytest.mark.parametrize("return_tensor", [False, True])
# @pytest.mark.parametrize("text", [
#     "Hello, world! This is a test of the tokenizer.",
#     "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
#     "Testing special characters: @#$%^&*()_+",
#     "",
# ])
# def test_tokenizer_single_text(text: str, return_tensor: bool) -> None:
#     run_tokenizer_case(text, return_tensor=return_tensor)

# @pytest.mark.parametrize("return_tensor", [False, True])
# def test_tokenizer_batch_text(return_tensor: bool) -> None:
#     texts = [
#         "Hello, world! This is a test of the tokenizer.",
#         "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
#         "Testing special characters: @#$%^&*()_+",
#     ]
#     run_tokenizer_case(texts, return_tensor=return_tensor, padding=True, truncation=True)

# @pytest.mark.parametrize("return_tensor", [False, True])
# def test_tokenizer_batch_text_no_truncation(return_tensor: bool) -> None:
#     texts = [
#         "Hello, world! This is a test of the tokenizer.",
#         "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
#         "Testing special characters: @#$%^&*()_+",
#     ]
#     run_tokenizer_case(texts, return_tensor=return_tensor, truncation=False)
                    
if __name__ == "__main__":
    
    # Test with a simple sentence
    text = "Hello, world! This is a test of the tokenizer."
    run_tokenizer_case(text, return_tensor = False)
    run_tokenizer_case(text, return_tensor = True)

    # Test with a longer text
    text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet."
    run_tokenizer_case(text, return_tensor = False)
    run_tokenizer_case(text, return_tensor = True)
    
    # Test with special characters
    text = "Testing special characters: @#$%^&*()_+"
    run_tokenizer_case(text, return_tensor = False)
    run_tokenizer_case(text, return_tensor = True)
    
    # Test with an empty string
    text = ""
    run_tokenizer_case(text, return_tensor = False)
    run_tokenizer_case(text, return_tensor = True)
    
    # test with a list with single sentence
    texts = ["Hello, world! This is a test of the tokenizer."]
    run_tokenizer_case(texts, return_tensor= False)
    run_tokenizer_case(texts, return_tensor = True)
    
    # test with a list of sentences
    texts = ["Hello, world! This is a test of the tokenizer.", 
             "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.", 
             "Testing special characters: @#$%^&*()_+"
            ]
    run_tokenizer_case(texts, return_tensor= False)
    run_tokenizer_case(texts, return_tensor = True)
    
    # test with a list of sentences with truncation = False
    texts = ["Hello, world! This is a test of the tokenizer.", 
             "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.", 
             "Testing special characters: @#$%^&*()_+"
            ]
    run_tokenizer_case(texts, return_tensor= False, truncation = False)
    run_tokenizer_case(texts, return_tensor = True, truncation = False)
    
    # test with a list of sentences with padding = False
    texts = ["Hello, world! This is a test of the tokenizer.", 
             "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.", 
             "Testing special characters: @#$%^&*()_+" 
            ]
    run_tokenizer_case(texts, return_tensor= False, padding = True)
    run_tokenizer_case(texts, return_tensor = True, padding = True)
    
    print("All tests passed!")