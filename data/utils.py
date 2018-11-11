import re
from data.constants import *

def normalize_string(s):
	s = s.lower()
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def get_idx_sentence(voc, word_sentence):
	return [voc.wrd2idx[word] for word in word_sentence.split(' ')] + [EOS_token]

def idx_to_sent(sent, dataset):
	res = ''
	for idx in range(sent.size(0)):
		word_idx = sent.cpu().numpy()[idx,0,0]
		if word_idx == 0:
			break
		else:
			res += dataset.vocab.idx2wrd[word_idx] + ' '

	return res