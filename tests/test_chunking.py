from utils.text import chunk_text, sentence_split

def test_sentence_split():
    text = "Hello world. This is a test! Is it working? Yes."
    sents = sentence_split(text)
    assert len(sents) == 4
    assert sents[0] == "Hello world."
    assert sents[1] == "This is a test!"

def test_chunk_text():
    text = "word " * 100
    # max 20, overlap 5
    chunks = chunk_text(text, max_words=20, overlap_words=5)
    assert len(chunks) > 0
    # Check length
    first = chunks[0].split()
    assert len(first) == 20
    # Check overlap
    second = chunks[1].split()
    # first[15:] should match second[:5] (approx, stride is 15)
    # Stride = 15. So chunk 2 starts at index 15.
    # Words 15,16,17,18,19 are in both chunk 0 and chunk 1.
    assert first[-5:] == second[:5]
