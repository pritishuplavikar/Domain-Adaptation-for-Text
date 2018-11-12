import torch

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
