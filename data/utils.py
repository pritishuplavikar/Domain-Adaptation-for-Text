import re
from data.constants import *

def normalize_string(s):
	s = s.lower()
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def get_idx_sentence(voc, word_sentence):
	return [voc.wrd2idx[word] for word in word_sentence.split(' ')] + [EOS_token]

def idx_to_sent(sent, vocab):
	res = ''
	for idx in range(sent.size(0)):
		word_idx = sent.cpu().numpy()[idx,0,0]
		if word_idx == 0:
			break
		else:
			res += vocab.idx2wrd[word_idx] + ' '

	return res

def get_addn_feats(preds, vocab):
	preds = preds.max(dim=1)[1]
	net_score = 0
	domain_A_count = 0
	domain_B_count = 0
	sent_len = 0
	for word in preds:
		word = word.item()
		if not word in vocab.tokens:
			sent_len += 1
			word = vocab.idx2wrd[word]
			if word in vocab.domain_A_vocab and word in vocab.domain_B_vocab:
				net_score += vocab.domain_A_vocab[word] - vocab.domain_B_vocab[word]
			elif word in vocab.domain_A_vocab:
				net_score += vocab.domain_A_vocab[word]
				domain_A_count += 1
			elif word in vocab.domain_B_vocab:
				net_score -= vocab.domain_B_vocab[word]
				domain_B_count += 1

	return torch.Tensor([net_score, domain_A_count, domain_B_count]).unsqueeze(0) / sent_len