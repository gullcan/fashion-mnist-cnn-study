import torch


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        batch_count = len(targets)
        loss_sum += loss.item() * batch_count
        correct_count += (logits.argmax(dim=1) == targets).sum().item()
        sample_count += batch_count

    mean_loss = loss_sum / sample_count
    accuracy = 100 * correct_count / sample_count

    return mean_loss, accuracy

def evaluate(model, loader, loss_fn, device):
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = loss_fn(logits, targets)

            batch_count = len(targets)
            loss_sum += loss.item() * batch_count
            correct_count += (logits.argmax(dim=1) == targets).sum().item()
            sample_count += batch_count

    mean_loss = loss_sum / sample_count
    accuracy = 100 * correct_count / sample_count

    return mean_loss, accuracy