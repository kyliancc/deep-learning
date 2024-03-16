import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from isicdataset import ISICDataset
from loss import UNetLoss
from unet import UNet
from logger import UNetLogger


def train(batch_size=8,
          lr=1e-5,
          max_epochs_to_train=10,
          validate_per_n=25,
          save_per_n=25,
          n_batches_to_val=-1,
          checkpoint_save_dir='./checkpoints',
          checkpoint_name_format='unet_checkpoint_{}.pth',
          checkpoint_load_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger
    logger = UNetLogger()
    logger.log('Initialized logger!')

    logger.log(f'''
    Properties:
        batch_size: {batch_size},
        lr: {lr},
        max_epochs_to_train: {max_epochs_to_train},
        validate_per_n: {validate_per_n},
        save_per_n: {save_per_n},
        n_batches_to_val: {n_batches_to_val}
        checkpoint_save_dir: {checkpoint_save_dir}
        checkpoint_name_format: {checkpoint_name_format}
        checkpoint_load_path: {checkpoint_load_path}
    ''')

    # Dataset & Dataloader
    train_set = ISICDataset('D:/Dataset/ISIC_2017_Task_1', usage='train')
    val_set = ISICDataset('D:/Dataset/ISIC_2017_Task_1', usage='val')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)

    logger.log('Loaded dataset!')

    # Model
    unet_model = UNet(3, 1, bilinear=True)
    unet_model.to(device=device)

    # Optimizer & Criterion
    optimizer = optim.RMSprop(unet_model.parameters(), lr=lr)
    criterion = UNetLoss()

    # Navigation parameters
    epoch_start = 1
    index_start = 0

    # Global stats
    iterations = 0

    # Load checkpoint
    if checkpoint_load_path is not None:
        checkpoint_data = torch.load(checkpoint_load_path)

        epoch_start = checkpoint_data['epoch']
        index_start = int(checkpoint_data['index'] * checkpoint_data['batch_size'] / batch_size) + 1

        if checkpoint_data['error']:
            logger.log('Detected this checkpoint was saved because of an error, trying to restore it...')
            index_start -= 1

        iterations = checkpoint_data['iterations']

        unet_model.load_state_dict(checkpoint_data['unet_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

        logger.log(f'Loaded checkpoint from {checkpoint_load_path}!')

    # Training loop
    for epoch in range(epoch_start, epoch_start + max_epochs_to_train):
        logger.log(f'Epoch {epoch}:')
        for index, data in enumerate(train_loader):
            error = False
            try:
                if index < index_start:
                    continue

                optimizer.zero_grad()

                img, mask = data
                img = img.to(device=device, dtype=torch.float32)
                mask = mask.to(device=device, dtype=torch.long)

                # Forward
                pred = unet_model(img)

                # Backward
                loss = criterion(pred, mask)
                loss.backward()
                optimizer.step()

                iterations += 1

                logger.log(f'Iteration {iterations} completed, running loss: {loss.item():.4f}')
                logger.log(f'Epoch progress: {((index + 1) / train_loader_len):.2%}')

                # Validation

                if iterations % validate_per_n == 0:
                    unet_model.eval()
                    with torch.no_grad():
                        val_loss_sum = 0.0
                        val_acc_sum = 0.0
                        for val_i, val_data in enumerate(val_loader):
                            if val_i == n_batches_to_val:
                                break

                            val_img, val_mask = val_data
                            val_img = val_img.to(device=device, dtype=torch.float32)
                            val_mask = val_mask.to(device=device, dtype=torch.long)

                            val_pred_soft = unet_model(val_img)
                            val_loss = criterion(val_pred_soft, val_mask)
                            val_loss_sum += val_loss.item()

                            val_pred_hard = (torch.sign(val_pred_soft) + 1) / 2
                            correct_pixels = torch.sum(torch.eq(val_pred_hard, val_mask), dtype=torch.float32)
                            total_pixels = torch.prod(torch.tensor(val_mask.size()), dtype=torch.float32)
                            val_acc_sum += correct_pixels / total_pixels

                        if n_batches_to_val > 0:
                            val_avg_loss = val_loss_sum / n_batches_to_val
                            val_avg_acc = val_acc_sum / n_batches_to_val
                        else:
                            val_avg_loss = val_loss_sum / val_loader_len
                            val_avg_acc = val_acc_sum / val_loader_len

                        logger.log(f'Validation completed with loss: {val_avg_loss:4f}, '
                                   f'accuracy: {val_avg_acc:.2%}')

                    unet_model.train()

            except Exception as e:
                error = True
                logger.log('Error: ' + str(e))

            # Save checkpoint
            if iterations % save_per_n == 0:
                checkpoint_data = {
                    'epoch': epoch,
                    'index': index,
                    'iterations': iterations,
                    'batch_size': batch_size,
                    'unet_state_dict': unet_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'error': error
                }
                checkpoint_path = checkpoint_save_dir + '/' + checkpoint_name_format.format(iterations)
                torch.save(checkpoint_data, checkpoint_path)
                logger.log(f'Saved checkpoint at {checkpoint_path}!')

            if error:
                exit(-1)


if __name__ == '__main__':
    train()
