import torch
from .vgg import Network
import torch.optim as optim

def load_model(type, PATH):
    model = Network(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    return model.eval()

def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)

def save_checkpoint(epoch, model, optimizer, loss, error, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'error': error
    }, PATH)

def load_checkpoint(PATH, training=True):
    model = Network(*args, **kwargs)
    optimizer = optim.SGD(model.parameters(), learningrate, momentum=momentum)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    error = checkpoint['error']

    if training:
        model.train()
    else:
        model.eval()

    return(model, optimizer, epoch, loss, error)

