from typing import Tuple
from tqdm import tqdm


def train(model, device, train_loader, optimizer, loss_fn) -> Tuple[float, float]:
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
   
        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = loss_fn(y_pred, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        # get the index of the max log-probability
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        accuracy = 100*correct/processed

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={accuracy:0.2f}')

    return accuracy, loss
