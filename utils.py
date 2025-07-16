arabic_chars = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأإىة،.؟ ")

char2idx = {ch: idx + 1 for idx, ch in enumerate(arabic_chars)}  # 0 = PAD
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(char2idx) + 1  # plus PAD

def encode_text(text, max_len=64):
    indices = [char2idx.get(c, 0) for c in text[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices
