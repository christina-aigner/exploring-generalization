import torch
import math
import copy


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


def calc_measure(model, init_model, measure_func, operator, kwargs={}, p=1):
    """
   calculates a measure on the given model measure_func is a function that returns
   a value for a given linear or convolutional layer
   calc_measure calculates the values on individual layers and then calculate
   the final value based on the given operation.

    """

    if operator == 'product':
        measure_val = math.exp(calc_measure(model, init_model, measure_func, 'log_product', kwargs, p))
    elif operator == 'norm':
        measure_val = (calc_measure(model, init_model, measure_func, 'sum', kwargs, p=p)) ** (1 / p)
    else:
        measure_val = 0
        for child, init_child in zip(model.children(), init_model.children()):
            module_name = child._get_name()
            if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
                if operator == 'log_product':
                    measure_val += math.log(measure_func(child, init_child, **kwargs))
                elif operator == 'sum':
                    measure_val += (measure_func(child, init_child, **kwargs)) ** p
                elif operator == 'max':
                    measure_val = max(measure_val, measure_func(child, init_child, **kwargs))
            else:
                measure_val += calc_measure(child, init_child, measure_func, operator, kwargs, p=p)
    return measure_val


def norm(module, p=2, q=2):
    """
    calculates l_pq norm of the parameter matrix of a layer:
    1) l_p norm of incoming weights to each hidden unit and l_q norm on the hidden units
    2) conv. tensors are reshaped s.t. all dimensions except the output are together

    """
    return module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()


def op_norm(module, p=float('Inf')):
    """
    calculates l_p norm of eigenvalues of a layer, convolutional tensors are reshaped
    s.t. all dimensions (except the output) are together

    """
    _, S, _ = module.weight.view(module.weight.size(0), -1).svd()
    return S.norm(p).item()


def distance(module, init_module, p=2, q=2):
    """
    calculates l_pq distance of the parameter matrix of a layer from the random
    initialization:
    1) l_p norm of incoming weights to each hidden unit and l_q norm on the hidden units
    2) conv. tensors are reshaped s.t. all dimensions (except output) are together

    """
    reshaped = (module.weight - init_module.weight).view(module.weight.size(0), -1)
    norm = reshaped.norm(p=p, dim=1).norm(q)
    return norm.item()


def h_dist(module, init_module, p=2, q=2):
    """
    calculates l_pq distance of the parameter matrix of a layer from the random
    initialization with an extra factor that depends on the number of hidden units.
    Args:
        module:
        init_module:
        p:
        q:

    Returns:

    """
    hidden = (n_hidden(module) ** (1 - 1 / q ))
    dist = distance(module, init_module, p=p, q=q)
    return hidden * dist


def h_dist_op_norm(module, init_module, p=2, q=2, p_op=float('Inf')):
    """
    calculates the ratio of the h_dist to the operator norm
    Args:
        module:
        init_module:
        p:
        q:
        p_op:

    Returns:

    """
    return h_dist(module, init_module, p=p, q=q) / op_norm(module, init_module, p=p_op)


def n_hidden(module, init_module):
    """
    Gets the number of hidden units
    Args:
        module:
        init_module:

    Returns:

    """
    return module.weight.size(0)


def depth(module, init_module):
    """
    Gets the depth of a module. --> always 1 for any linear of convolutional layer

    Args:
        module:
        init_module:

    Returns:

    """

    return 1


def n_param(module, init_module):
    """
    Gets the number of parameters of a module.
    Args:
        module:
        init_module:

    Returns:
    """
    bparam = 0 if module.bias is None else module.bias.size(0)
    return bparam + module.weight.size(0) * module.weight.view(module.weight.size(0),-1).size(1)


def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    """
    calculates the path-norm after Neyshabur et al. 2015
    Args:
        model:
        device:
        p:
        input_size:

    Returns:

    """
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p)).item()
