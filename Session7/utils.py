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


def get_misclassified_images(gbn_model,test_loader):
    
    test_images = []
    target_labels = []
    target_predictions = []
    for img, target in test_loader:
      prediction = torch.argmax(gbn_model(img), dim=1)
      test_images.append( img )
      target_labels.append( target )
      target_predictions.append( prediction )

    test_images = torch.cat(test_images)
    target_labels = torch.cat(target_labels)
    target_predictions = torch.cat(target_predictions)
    misclassified_index = target_labels.ne(target_predictions).numpy()
    test_images = test_images[misclassified_index]
    target_labels = target_labels[misclassified_index]
    target_predictions = target_predictions[misclassified_index]

    return test_images,target_labels,target_predictions
