import math

def adjust_learning_rate(n_epoch_warmup, n_epoch, max_lr, optimizer, loader, step):
    # Calculate the total number of training steps
    max_steps = int(n_epoch * len(loader))
    # Calculate the number of warmup steps
    warmup_steps = int(n_epoch_warmup * len(loader))
    
    # If we are in the warmup phase
    if step < warmup_steps:
        # Linear warmup
        lr = max_lr * step / warmup_steps
    # If we are in the decay phase
    else:
        # Subtract warmup steps from step and max_steps
        step -= warmup_steps
        max_steps -= warmup_steps
        
        # Cosine decay
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        # Calculate the end learning rate
        end_lr = max_lr * 0.001
        # Calculate the current learning rate
        lr = max_lr * q + end_lr * (1 - q)
    # Update the learning rate in the optimizer
    optimizer.param_groups[0]['lr'] = lr


def adjust_sigma(n_epoch_warmup, n_epoch, max_sigma, DPS, loader, step):
    # Calculate the total number of training steps
    max_steps = int(n_epoch * len(loader))
    # Calculate the number of warmup steps
    warmup_steps = int(n_epoch_warmup * len(loader))
    
    # If we are in the warmup phase
    if step < warmup_steps:
        # Linear warmup
        sigma = max_sigma * step / warmup_steps
    # If we are in the decay phase
    else:
        # Subtract warmup steps from step and max_steps
        step -= warmup_steps
        max_steps -= warmup_steps
        
        # Cosine decay
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        # Calculate the end sigma value
        end_sigma = 1e-5
        # Calculate the current sigma value
        sigma = max_sigma * q + end_sigma * (1 - q)
    # Update sigma in the DPS module
    DPS.TOPK.sigma = sigma
