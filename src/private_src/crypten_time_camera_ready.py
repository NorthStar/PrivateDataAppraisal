# -*- coding: utf-8 -*-
"""crypten_time_camera_ready.py

Planecar_app.py in crypten. Timing only.

Before influence, encrypt the inverse of the hessian
encrypt the candidate sets. Tensor level, decrypt the result.

Inverse hessian product and trained models are provided.

@args
LOAD_DATA_FORMAT        Saved key data from PlaneToCar App.
LOAD_AUX_DATA_FORMAT    Saved aux data from PlaneToCar App.
USE_CRYPTEN             Whether to run encrypted timing.

truncation              The number of data sets to evaluate.

The paper uses ecrypted timing, subtracting overhead of tensor loading.

@remarks
Crypten model initialization is very similar to PyTorch's.
Ex1: nn.Linear (extract weights, module)
Ex2: loss (nn.Linear-> crypten.nn.Linear )

Sensitivity: Precision set to 24

"""
# Required pre-train model struct
import crypten
import time
import torch

import crypten.nn as cnn
import torch.nn as tnn
import torchvision.datasets as dss
import torchvision.transforms as transforms

DIMS = 3 * 32 * 32
NUM_CLASSES = 2
NUM_RUNS = 5

# TODO (mimee): config file
WEIGHT_DECAY = 1e-3
LOAD_DATA_FORMAT = ""
LOAD_AUX_DATA_FORMAT = ""

## TODO: import these
class timeit:
    # timing utils
    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        print(f"{interval_ms:8.2f}   {self.tag}")

class cache_tensor:
    def __init__(self, tag=""):
        self.tag = tag
    def cache_obj(self, obj):
        torch.save(obj, self.tag + ".pt")
    def load_obj(self):
        return torch.load(self.tag + ".pt")

def print_tensor(tt, use_crypten=False):
    if use_crypten:
        print(tt.get_plain_text())
    else:
        print(tt)

def onehot(indices, num_targets=None):
    """
    To avoid
    `AssertionError: input and target must have the same size`
    Converts index vector into one-hot matrix.

    """
    assert indices.dtype == torch.long, "indices must be long integers"
    assert indices.min() >= 0, "indices must be non-negative"
    if num_targets is None:
        num_targets = indices.max() + 1
    onehot_vector = torch.zeros(indices.nelement(), num_targets, dtype=torch.long)
    onehot_vector.scatter_(1, indices.view(indices.nelement(), 1), 1)
    return onehot_vector

def ce_loss(x,t, use_crypten=False):
    if not use_crypten:
        return tnn.CrossEntropyLoss()(x, t)
    #return x.log_softmax(1,).dot(t).neg() # unstable
    return cnn.CrossEntropyLoss()(x,t)

def data_load(dims=DIMS):
    all_data = cache_tensor(LOAD_DATA_FORMAT.format(DIMS)).load_obj()
    aux_data = cache_tensor(LOAD_AUX_DATA_FORMAT.format(DIMS)).load_obj()
    return (all_data, aux_data)

def contained_linear_layer_from_weights(layerweight, use_crypten=False):
    """Creates simple linear model with pytorch container."""
    if not use_crypten:
        layer = tnn.Linear(DIMS, NUM_CLASSES, bias=False)
    else:
        layer = cnn.Linear(DIMS, NUM_CLASSES, bias=False)

    layer.weight.copy_(layerweight)
    layer.weight.grad = None
    return layer

def uncontained_linear_layer_from_weights(layerweight):
    """Creates simple linear model.
    returns func(x):W X^T where X is a whole batch.
    This already works if BOTH layerweight and input are cryptensors."""
    return (lambda x: layerweight @ x.t())

def model_from_weights(layerweight, dims=DIMS, num_classes=NUM_CLASSES, use_crypten=False):
    if not use_crypten:
        layer = tnn.Linear(dims, num_classes , bias=False)
        with torch.no_grad():
            layer.weight.copy_(layerweight)
        return layer

    layer_crypten = cnn.Linear(dims, num_classes , bias=False)

    # Copy over weights in plain text before encrypting the layer.
    with crypten.no_grad():
        layer_crypten.weight = layerweight.clone()
    layer_crypten.encrypt()
    return layer_crypten

def c_L_d_theta_regularized(data, target, layerweight):
    """Purely functional implementation for single layer model."""
    try:
        layerweight.grad.data._zero()
    except:
        pass

    return torch.autograd.functional.jacobian(
        lambda we: ce_loss((we @ data.t()).t(), target) + WEIGHT_DECAY * (we**2).sum()/2,
        layerweight,
        create_graph=True,
        strict=True
    )

def L_d_theta(data, target, layerweight, use_crypten=False):
    """Compute in secret or in the plain the gradient on the parameters.
    Purely functional implementation for single layer model.
    """
    # construct the model and run backwards
    #print("L_d_theta use_crypten={}".format(use_crypten))
    model = model_from_weights(layerweight, use_crypten=use_crypten)
    try:
        layerweight.grad.data.zero_() # zero the gradients
    except:
        pass

    if not use_crypten:

        reg = WEIGHT_DECAY * (layerweight**2).sum() / 2
        # jacobian of weight_decay * (w**2).sum() / 2 on w = weight_decay * w.sum()

        loss = ce_loss(model(data.view(-1, DIMS)), target) + reg
        with timeit("PLAIN Taking gradient for influence"):
            loss.backward()
            return model.weight.grad.data

    with timeit("INF overhead data tensor load"):
        data_crypten = crypten.cryptensor(data.view(-1, DIMS))
        target_crypten = crypten.cryptensor(onehot(target, num_targets=NUM_CLASSES))

    with timeit("INF: write loss, take encrypted gradient"):
        loss = ce_loss(model(data_crypten), target_crypten, use_crypten=use_crypten) + crypten.cryptensor(WEIGHT_DECAY * (layerweight**2).sum() / 2)

        loss.backward()
        return model.weight.grad

def compute_gd(weights, data, targets, test_data, test_targets, use_crypten, weight_decay=1e-3):
    """Times gradient norm."""
    print("gradient norm use_crypten={}".format(use_crypten))
    model = model_from_weights(weights, use_crypten=use_crypten)
    try:
        weights.grad.data.zero_() # zero the gradients
    except:
        pass

    #with timeit("TOTAL GD: ENC GD + setup"):
        with timeit("GD overhead: Initialize tensors."):
            data_crypten = crypten.cryptensor(data.view(-1, DIMS))

            target_crypten = crypten.cryptensor(onehot(targets, num_targets=NUM_CLASSES))

        model.zero_grad()
        with timeit("GD: Write loss, encrypt gradient, take norm and return"):
            loss = ce_loss(model(data_crypten), target_crypten, use_crypten=use_crypten) + crypten.cryptensor(WEIGHT_DECAY * (weights**2).sum() / 2)

            loss.backward()
            return model.weight.grad.norm()
        # print(f"ENCRYPTED gd norm {gd_norm}")

def compute_inf(add_train_data, add_train_target, Lte_H_inv, layerweight, use_crypten=False):
    # print("compute_inf use_crypten={}".format(use_crypten))
    Lat = L_d_theta(add_train_data, add_train_target, layerweight, use_crypten=use_crypten)

    try:
        layerweight.grad.data._zero()
    except:
        pass

    if not use_crypten:
        return - (Lte_H_inv @ torch.flatten(Lat)).data
    # Flatten does not work on encrypted data.
    with timeit("INF 1: matmul norm, batch_inf_after_reinitializing_model"):
        return - (crypten.cryptensor(Lte_H_inv) @ Lat.reshape(-1))


def batch_compute_inf(
    Lte_H_inv, train_data, train_target,
    test_data, test_target, add_train_set,
    layerweight, use_crypten=False):
    """Outputs influence with given weights and additional data.
    $$Negative inf = \nabla L(x_test,\theta) ^T \@ H^{-1} @ \nabla L (x_{add_train})$$. """
    # Take the first layer weights and compute grad, hessian wrt to it.
    try:
        layerweight.grad.data._zero()
    except:
        pass

    with timeit("TOTAL INF: batched"):
        infs = [compute_inf(add_train_data, add_train_target, Lte_H_inv, layerweight, use_crypten=use_crypten) for (add_train_data, add_train_target) in add_train_set]
        return infs


def fine_tune(weights, data, targets, test_data, test_targets, epochs=20, learning_rate=0.001, weight_decay=1e-3, use_crypten=False, full_batch=True):
    """Finetune with given epoches and learning rate.

    TODO: Use crypten.optim
    TODO: Use batches
    TODO: Compare weights
    TODO: Use timing comparisons

    TODO: check test loss
    """

    print("fine_tune use_crypten={}".format(use_crypten))
    model = model_from_weights(weights, use_crypten=use_crypten)
    try:
        weights.grad.data.zero_() # zero the gradients
    except:
        pass

    if not use_crypten:
        with torch.no_grad():
            # record the before loss
            reg = weight_decay * (model.weight**2).sum()/2
            before_loss = ce_loss(model(test_data), test_targets) + reg
            print(f"PLAIN before loss: {before_loss.data}")

    if (not use_crypten) and (not full_batch):
        print("Using optim SGD:")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # batch_size = 1
        for epoch in range(epochs):
            for i in range(len(data)):
                image = data[i]
                tt = targets[[i]]
                # safer than optimizer.zero_grad()
                model.zero_grad()

                loss = ce_loss(model(image.view(-1, DIMS)), tt)
                loss.backward()
                if epoch % 50 == 0:
                    print(f"PLAIN epoch {epoch} train loss: {loss}")
                optimizer.step() # includes L2 regularizer

        reg = weight_decay * (model.weight**2).sum()/2
        after_loss = ce_loss(model(test_data), test_targets) + reg
        return (after_loss.data, model.weight.data)

    if (not use_crypten) and full_batch:
        print("Hand-written update param")
        for epoch in range(epochs):
            model.zero_grad()
            loss = ce_loss(model(data.view(-1, DIMS)), targets) #+ weight_decay * (model.weight**2).sum()/2
            loss.backward()
            print(f"PLAIN epoch {epoch} loss: {loss}")
            #optimizer.step()
            with torch.no_grad():
                #weights = weights - learning_rate * model.weight.grad
                weights = weights * (1 - weight_decay*learning_rate) - learning_rate * model.weight.grad
                model.weight.data = weights.clone()
        reg = weight_decay * (model.weight**2).sum()/2
        after_loss = ce_loss(model(test_data), test_targets) + reg
        return (after_loss.data, model.weight.data)#model.weight.grad.data)


    #optimizer_crypten = crypten.optim.SGD(
    #    model.parameters(), lr=learning_rate, weight_decay=weight_decay
    #)

    # Use crypten.
    with timeit("FT Overhead: initialize data"):
        data_crypten = crypten.cryptensor(data.view(-1, DIMS))
        target_crypten = crypten.cryptensor(onehot(targets, num_targets=NUM_CLASSES))
    print("Output tensor encrypted:", crypten.is_encrypted_tensor(model(data_crypten)))
    print("dim check output:{}, target: {}".format(model(data_crypten).shape, target_crypten.shape))

    # full batch descent
    print('FT runs {} epochs'.format(epochs))
    with timeit("FT: Full batch, no reg"):
        for epoch in range(epochs):
            model.zero_grad()

            # Because there are no optimizer implementation with weight decay yet,
            # the weight decay needs to be put in the loss explicitly
            # For timing, we omit that as to
            # not penalize under-optimized finetuning implementation
            loss = ce_loss(
                    model(data_crypten),
                    target_crypten,
                    use_crypten=use_crypten
                )

            loss.backward()

            model.update_parameters(learning_rate)  # just GD

    print_tensor(model.weight.grad, use_crypten)

    # recall that model.weight is a cryptensor already
    after_loss = ce_loss(
        model(crypten.cryptensor(test_data)),
        crypten.cryptensor(onehot(test_targets)),
        use_crypten=use_crypten
        )

    return (after_loss.get_plain_text(), model.weight.get_plain_text())


if __name__ == "__main__":
    from crypten.encoder import FixedPointEncoder
    crypten.init()

    crypten.encoder.precision_bits = 24
    print("crypting!!")

    # Prep data from cache.
    (train_data, train_targets, train_num, test_data, test_targets, test_num, additional_training_sets, additional_set_num), (weights, Lte_H_inv) = data_load()

    # reshape if needed
    test_data = test_data.reshape(test_num, DIMS)

    Lte_H_inv = Lte_H_inv/10000/100
    use_crypten = True # Use this as an input.

    # Truncate
    additional_set_num = 100
    additional_training_sets = additional_training_sets[:additional_set_num]

    with timeit("compute inf crypten AFTER loading tensor in memory"):
        crypten_infs = batch_compute_inf(
            Lte_H_inv,
            train_data,
            train_targets,
            test_data,
            test_targets,
            additional_training_sets,
            weights,
            use_crypten=True
        )
    print("INF_ENCRYPTED={}".format([inf.get_plain_text() for inf in crypten_infs]))

    [print_tensor(inf, use_crypten=True) for inf in crypten_infs]


    with timeit("compute inf plain"):
        plain_infs = batch_compute_inf(
            Lte_H_inv,
            train_data,
            train_targets,
            test_data,
            test_targets,
            additional_training_sets,
            weights,
            use_crypten=False
        )
    print("INF_PLAIN={}".format([inf for inf in plain_infs]))
    [print_tensor(inf, use_crypten=False) for inf in plain_infs]

    def plain_loss_from_weights(plain_weights):
        """Computes loss from weights."""
        plain_model = model_from_weights(plain_weights, use_crypten=False)
        plain_reg = WEIGHT_DECAY * (plain_weights**2).sum()/2
        plain_after_loss = ce_loss(plain_model(test_data), test_targets, use_crypten=False) + plain_reg
        return plain_after_loss.data

    ## Perform finetuning
    def batch_finetune(use_crypten=False, full_batch=True):
        ft_res = []
        for ats, ats_targets in additional_training_sets:
            total_data = torch.cat([train_data.view(-1, DIMS), ats.view(-1, DIMS)])
            total_targets = torch.cat([train_targets, ats_targets])

            ft_res.append(
                fine_tune(
                    weights, total_data, total_targets, test_data, test_targets,
                    epochs=16, learning_rate = 0.01, use_crypten=use_crypten, full_batch=full_batch)
            )
        return ft_res

    ## Timing GD in Crypten only
    def batch_gd(use_crypten=True):
        for ats, ats_targets in additional_training_sets:
            total_data = torch.cat([train_data.view(-1, DIMS), ats.view(-1, DIMS)])
            total_targets = torch.cat([train_targets, ats_targets])

            compute_gd(weights, total_data, total_targets, test_data, test_targets, use_crypten=use_crypten)

    # Performance
    with timeit("compute finetune plain f_b"):
        plain_ft_res = batch_finetune(use_crypten=True, full_batch=True)
    print("FT_PLAIN [full batch GD no optim]: {}".format([loss_data for (loss_data, _) in plain_ft_res]))

    # ## Comment this out to run FT, LFT experiments.
    with timeit("compute finetune encrypted"):
        crypten_ft_res = batch_finetune(use_crypten=True)

    # If we want to compare resulting tensors.
    # print("FT_ENCRYPTED [full batch GD]: {}".format([loss_data for (loss_data, _) in crypten_ft_res]))
    # print("UNENCRYPTED_LOSSES:{}".format([plain_loss_from_weights(pw) for (_, pw) in crypten_ft_res]))

    with timeit("compute gd crypten"):
        batch_gd()

    exit(0)
