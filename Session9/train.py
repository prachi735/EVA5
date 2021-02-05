from tqdm import tqdm
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc, loss_fn):

  model.train()
  pbar = tqdm(train_loader)
  train_loss = 0
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    
    y_pred = model(data)

    # Calculate loss
    loss = loss_fn(y_pred, target)
    train_losses.append(train_loss)  # .item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    # get the index of the max log-probability
    pred = y_pred.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(
        desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
    
    print('\Train set: Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
