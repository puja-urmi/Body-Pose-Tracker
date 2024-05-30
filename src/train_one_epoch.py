from tqdm import tqdm

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"epoch: {epoch_index}"):
        inputs, labels = data['image'], data['kp']
        inputs, labels  = inputs, labels
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    last_loss = running_loss / len(train_loader)
    print('  batch {} loss: {}'.format(i + 1, last_loss))
    tb_x = epoch_index * len(train_loader) + i + 1
    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    running_loss = 0.

    return last_loss