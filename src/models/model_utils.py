import torch
from .vgg import Network
import torch.optim as optim


def load_model(PATH, nchannels=3, nclasses=10):
    model = Network(nchannels, nclasses)
    if not torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()


def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)


def save_checkpoint(epoch, model, optimizer, random_labels, tr_loss, tr_error, val_error, margin, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'random_labels': random_labels,
        'optimizer_state_dict': optimizer.state_dict(),
        'tr_loss': tr_loss,
        'tr_error': tr_error,
        'val_error': val_error,
        'margin': margin
    }, PATH)


def load_checkpoint_eval(PATH, nchannels=3, nclasses=10):

    checkpoint = torch.load(PATH)
    model = Network(nchannels, nclasses)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def load_checkpoint_train(PATH, nchannels=3, nclasses=10, learningrate=0.01, momentum=0.9):

    model = Network(nchannels, nclasses)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.SGD(model.parameters(), learningrate, momentum=momentum)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.train()

    return model, optimizer, checkpoint

