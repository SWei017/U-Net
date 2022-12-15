import torch.optim as optim


def load_optimizer(opt, model):
    # Return optimizer based on options
    if opt.optimizer == 'Adamax':
        return optim.Adamax(model.parameters(), opt.lr)
    if opt.optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), opt.lr)


def load_scheduler(opt, optimizer):
    # Return scheduler based on options
    if opt.scheduler == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    assert 'Scheduler does not exist'

