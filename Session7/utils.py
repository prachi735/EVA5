SEED = 3

# is cuda available
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

torch.manual_seed(SEED)

if cuda:
  torch.cuda.manual_seed(SEED)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def run_model(model, device, optimiser, EPOCHS=1, is_L1_loss=False, is_GBN=False, gbn_splits=2):

  train_losses = []
  train_acc = []
  test_losses = []
  test_acc = []

  for epoch in range(EPOCHS):
      print("EPOCH:", epoch+1)
      train(model, device, train_loader, optimizer, epoch,
            train_losses, train_acc, is_L1_loss, lamda_l1=0.0001)
      test(model, device, test_loader, test_losses, test_acc)

  return {'train_loss': train_losses,  'train_acc': train_acc,  'test_loss': test_losses,  'test_acc': test_acc}
