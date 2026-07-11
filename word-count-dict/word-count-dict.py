def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    word_count = {}
    for sentence in sentences:
        for token in sentence:
            word_count[token] = word_count.get(token, 0)+ 1
    return word_count