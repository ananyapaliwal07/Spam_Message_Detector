def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]','',text)
    words = text.split()
    new_words = []
    for w in words:
        if w not in stop_words:
            new_words.append(w)
    words = new_words
    return " ".join(words)