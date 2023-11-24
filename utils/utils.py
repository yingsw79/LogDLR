import time


def timer(func):

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        res = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'{func.__name__} took {hours:.0f} hours, {minutes:.0f} minutes and {seconds:.2f} seconds to execute.\n')
        return res

    return wrapper


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
