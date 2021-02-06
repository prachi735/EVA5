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


def plot_sample_imagesdataloader(dataloader, num=5, fig_size=(10, 10)):
  dataiter = iter(dataloader)

  images, labels = dataiter.next()

  figure = plt.figure(figsize=fig_size)
  num_of_images = num
  for index in range(1, num_of_images + 1):
      plt.subplot(5, 5, index)
      plt.axis('off')
      plt.imshow(images[index][0])


def get_misclassified_images(model, test_loader, count=25):

    img_count = 0
    test_images = []
    target_labels = []
    target_predictions = []
    for img, target in test_loader:
      prediction = torch.argmax(model(img), dim=1)
      test_images.append(img)
      target_labels.append(target)
      target_predictions.append(prediction)
      img_count += 1
      if img_count == count:
        break


    test_images = torch.cat(test_images)
    target_labels = torch.cat(target_labels)
    target_predictions = torch.cat(target_predictions)
    misclassified_index = target_labels.ne(target_predictions).numpy()
    test_images = test_images[misclassified_index]
    target_labels = target_labels[misclassified_index]
    target_predictions = target_predictions[misclassified_index]

    return test_images, target_labels, target_predictions


def plot_results(train_losses, train_acc, test_losses, test_acc):
    data = {'train_loss': train_losses,  'train_acc': train_acc,
            'test_loss': test_losses,  'test_acc': test_acc}
    fig, axs = plt.subplots(1, 4, figsize=(30, 5))
    axs_pos = {'train_loss': (0),
               'train_acc': (1),
               'test_loss': (2),
               'test_acc': (3)}

    for i in data:
        ax = axs[axs_pos[i]]
        ax.plot(data[i])
        ax.set_title(i)


def show_misclassified_images(test_images, target_labels, target_predictions, classes, nrow=5, ncol=5, fig_size=(10, 10)):
  fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=fig_size)
  fig.subplots_adjust(hspace=0.5)
  fig.suptitle('Misclassified Images in Model')
  for ax, image, target, prediction in zip(axes.flatten(), test_images, target_labels, target_predictions):
      ax.imshow(image[0])
      ax.set(title='target:{t} prediction:{p}'.format(
          t=classes[target.item()], p=classes[prediction.item()]))
      ax.axis('off')
