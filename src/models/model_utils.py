import torch
import torch.optim as optim

from models import vgg, fc


def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)


def save_checkpoint(epoch, model, optimizer, random_labels, tr_loss, tr_error, val_error, margin,
                    PATH, sharpness=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'random_labels': random_labels,
        'optimizer_state_dict': optimizer.state_dict(),
        'tr_loss': tr_loss,
        'tr_error': tr_error,
        'val_error': val_error,
        'margin': margin,
        'sharpness': sharpness
    }, PATH)


def load_model(PATH, network='vgg', hiddenunits=1024, nchannels=3, nclasses=10):
    if network == 'vgg':
        model = vgg.Network(nchannels, nclasses)
    elif network == 'fc':
        model = fc.Network(hiddenunits, nchannels, nclasses)
    else:
        raise ValueError("no valid network parameter.")

    if not torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()


def load_checkpoint_dict(PATH):
    if not torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(PATH)
    return checkpoint


def load_checkpoint_train(PATH, network='vgg', hiddenunits=1024, nchannels=3,
                          nclasses=10, learningrate=0.01, momentum=0.9):
    if not torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(PATH)

    if network == 'vgg':
        model = vgg.Network(nchannels, nclasses)
    elif network == 'fc':
        model = fc.Network(hiddenunits, nchannels, nclasses)
    else:
        raise ValueError("no valid network parameter.")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.SGD(model.parameters(), learningrate, momentum=momentum)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.train()

    return model, optimizer, checkpoint


def reparam(model, prev_layer=None):
    """
    This function reparametrizes the networks with batch normalization in a way that it
    calculates the same function as the original network but without batch normalization.
    Instead of removing batch norm completely, we set the bias and mean to zero, and
    scaling and variance to one.
    Warning: This function only works for convolutional and fully connected networks.
    It also assumes that module.children() returns the children of a module in
    forward pass order. Recursive construction is allowed.

    """
    for child in model.children():
        module_name = child._get_name()
        prev_layer = reparam(child, prev_layer)
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            prev_layer = child
        elif module_name in ['BatchNorm2d', 'BatchNorm1d']:
            with torch.no_grad():
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.copy_( child.bias + (scale * (prev_layer.bias - child.running_mean)))
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
    return prev_layer