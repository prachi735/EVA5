import torch

def get_device():
  SEED = 3

  # is cuda available
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  torch.manual_seed(SEED)

  if cuda:
    torch.cuda.manual_seed(SEED)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device
