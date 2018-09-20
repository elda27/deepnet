from deepnet.core.registration import register_process

operation_list = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x / y,
}

@register_process()
def reduce(*input, operation='+', weights=None):
    if weights is None:
        weights = [ 1.0 for _ in range(len(input)) ]

    operation = operation_list[operation]
    input_iter = iter(zip(input, weights))

    x0, w0 = next(input_iter)
    y = x0 * w0
    for x, w in input_iter:
        y = operation(x * w, y)

    return y