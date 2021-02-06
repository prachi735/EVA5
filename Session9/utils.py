import torch
import matplotlib.pyplot as plt

def seed_torch():
  SEED = 3

  # is cuda available
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  torch.manual_seed(SEED)

  if cuda:
    torch.cuda.manual_seed(SEED)



def get_device():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device


def get_dataloader(data, ** dataloader_args):
  
  dataloader = torch.utils.data.DataLoader(data, ** dataloader_args)

  return dataloader


def plot_sample_imagesdataloader(dataloader, num=5, fig_size=(10, 10)):
  dataiter = iter(dataloader)

  images, labels = dataiter.next()

  figure = plt.figure(figsize=fig_size)
  num_of_images = num
  for index in range(1, num_of_images + 1):
      plt.subplot(5, 5, index)
      plt.axis('off')
      plt.imshow(images[index][0])
