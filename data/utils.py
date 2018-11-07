import re
from constants import *

def normalize_string(s):
	s = s.lower()
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def get_idx_sentence(voc, word_sentence):
	return [voc.wrd2idx[word] for word in word_sentence.split(' ')] + [EOS_token]