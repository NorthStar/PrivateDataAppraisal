# -*- coding: utf-8 -*-
"""planecar_app.py

For given list of epoches, run in plaintext
gradient norms, finetuning, and influence.

Report their times and correlation
with ground truth, ran through L-BFGS.

The results are stored in a CSV.

"""
import torch
import torchvision
import torchvision.transforms as transforms
import time

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import random

from torch.autograd import functional
from torch import autograd

from scipy import stats

import pandas as pd

from absl import app
from absl import flags
from absl import logging

import sys

FLAGS = flags.FLAGS

flags.DEFINE_float('weight_decay', 1e-3, 'weight_decay')
flags.DEFINE_integer('runs', 1, 'number of runs')
flags.DEFINE_list('e_sgds', [1], "list of number of epochs e.g. 1,2,4 ")
flags.DEFINE_float('lr_sgd', 0.1, "sgd learning rate")


weight_decay = 1e-3
DIMS = 3*32*32
add_train_size = 440
IMBALANCE_NUM = 20
IMBALANCE_REP = 5

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## HELPER
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

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

criterion = nn.CrossEntropyLoss()

def train_lbfgs(model, data, target, epochs, lbfgs_lr):
    """Forward step with lbfgs, full batch impl.
    """
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lbfgs_lr,
        tolerance_grad=1e-4,
        line_search_fn="strong_wolfe"
    )

    weights = model.weight
    for n in range(epochs):
            add_train = data
            add_target = target
            def loss_closure():
                optimizer.zero_grad()
                reg = weight_decay * (weights**2).sum()/2
                loss = nn.CrossEntropyLoss()(model(add_train), add_target) + reg
                loss.backward()
                return loss
            optimizer.step(closure=loss_closure)

weight_decay = 1e-3

def test_prep():
    train_data = torch.stack([transform(td) for td in trainset.data])
    train_targets = torch.stack([torch.tensor(tt) for tt in trainset.targets])

    test_data = torch.stack([transform(td) for td in testset.data])
    test_targets = torch.stack([torch.tensor(tt) for tt in testset.targets])

    t_idx = [idx for idx in range(len(testset)) if (testset.targets[idx] == 0 or testset.targets[idx] == 1)]
    t_data = test_data[t_idx]
    t_targets = test_targets[t_idx]

    # Make train set unbalanced
    tr0_idx = [idx for idx in range(len(train_targets)) if ((train_targets[idx] == 0) and (idx %2 == 0))]
    tr1_idx = [idx for idx in range(len(train_targets)) if ((train_targets[idx] == 1) and (idx %20 == 0))]


    batch_idx = random.sample(tr0_idx, 200) + random.sample(tr1_idx, 20)
    random.shuffle(batch_idx)
    batch_d, batch_t = train_data[batch_idx].reshape(len(batch_idx), 3*32*32), train_targets[batch_idx]

    nnet = nn.Linear(3*32*32, 2, bias=False)

    with timeit("compute pre-train n_net"):
        pretrain(nnet, batch_d, batch_t)

    before_loss = test_loss(nnet, t_data, t_targets)

    # Heldout, not in train, 0:1 = 19x
    ho_idx = [idx for idx in range(len(train_targets)) if (((train_targets[idx] == 0) and (idx %2 != 0)) or ((train_targets[idx] == 1) and (idx %20 != 0)))]

    ho0_idx = [idx for idx in range(len(train_targets)) if ((train_targets[idx] == 0) and (idx %2 != 0)) ]
    print("0-label has {} samples".format(len(ho0_idx)))
    ho1_idx = [idx for idx in range(len(train_targets)) if ((train_targets[idx] == 1) and (idx %20 != 0)) ]
    print("1-label has {} samples".format(len(ho1_idx)))

    ho_data = train_data[ho_idx]
    ho_targets = train_targets[ho_idx]

    unsorted_candidates = prep_candidates(
        train_data,
        train_targets,
        ho0_idx,
        ho1_idx,
        IMBALANCE_NUM,
        IMBALANCE_REP
    )

    return ({
        "pretrained_model": nnet,
        "training": (batch_idx, batch_d, batch_t),
        "testing": (t_idx, t_data, t_targets),
        "candidates": unsorted_candidates,
        "before_loss": before_loss
    })

def test_loss(model, test_images, test_targets):
  with torch.no_grad():
    reg = weight_decay * (model.weight**2).sum()/2
    loss = nn.CrossEntropyLoss()(model(test_images.reshape(len(test_images), 3*32*32)), test_targets) + reg
    return loss

def test_accuracy(model, t_data, t_targets):
  with torch.no_grad():
    outputs = model(t_data.reshape(len(t_data), 3*32*32))
    _, predicted = torch.max(outputs.data, 1)
    total = len(t_targets)
    correct = (predicted == t_targets).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))
  return correct / total


def class_accuracy(model, t_idx):
    #override
    testloader = torch.utils.data.DataLoader(
        [testset[idx] for idx in t_idx],
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    class_correct = [0,0]
    class_total = [0,0]
    with torch.no_grad():
        for data in testloader:
            images = data[0].reshape(4, 3*32*32)#.cuda()
            labels = data[1]#.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return (class_correct[0] / class_total[0], class_correct[1] / class_total[1])

def pretrain(model, batch_d, batch_t):
  train_lbfgs(model, batch_d, batch_t, 1001, 0.00001)

def functional_nnet(train_data, train_target):
  criterion = nn.CrossEntropyLoss()
  def dummy_loss(we):
    dummy_model = nn.Linear(3*32*32, 2, bias=False)
    with torch.no_grad():
        dummy_model.weight.copy_(we)
#    dummy_model.weight.grad = None
    reg = 1e-3 * (we**2).sum()/2
    return criterion(dummy_model(train_data), train_target) + reg
  return dummy_loss

def c_grad_curry_regularized(data, target):
  """A closure constructor with regularization term for functional."""
  def loss(layerweight):
    model = (lambda x: layerweight @ x.t())
    reg = 1e-3 * (layerweight**2).sum()/2
    return criterion(model(data).t(), target) + reg
  return loss

def prep_candidates(train_data, train_targets, ho0_idx, ho1_idx, samples=10, rep=1, noise=False):
  candidate_sets = []
  noise_fraction = 0
  for _ in range(rep):
    for i in range(samples):
      len0 = int(add_train_size * (samples - 1 -i )/samples)
      len1 = int(add_train_size * (i + 1) /samples)
      candidate_idx = random.sample(ho0_idx, len0) + random.sample(ho1_idx, len1)
      random.shuffle(candidate_idx)

      ho_tr = train_data[candidate_idx].reshape(add_train_size, 3*32*32)
      ho_tt = train_targets[candidate_idx]

      print("{}-th sample \n".format(i),len0, " samples are of label 0.\n", len1, " samples are of label 1.")
      candidate_sets.append((ho_tr, ho_tt, len0/(len0 + len1), noise_fraction))
  return candidate_sets

def train_sgd(model, data, target, epochs, sgd_lr, weight_decay, e_sgds=[1]):
    """Forward step with sgd, using batch=1.
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=sgd_lr,
        weight_decay=weight_decay
    )
    weights = model.weight

    # shuffle data
    idx = torch.randperm(len(data))
    data = data[idx]
    target = target[idx]

    # Train all the way till epoch_max, but store intermediate weights.
    resulting_weights = []
    e_idx = 0
    for n in range(epochs):
        # one sample at a time
        for i in range(len(data)):
            add_train = data[i].reshape(-1, DIMS) # crucial reshape
            add_target = target[[i]] # still need to be in a batch
            optimizer.zero_grad()

            loss = nn.CrossEntropyLoss()(model(add_train), add_target) # reg is automatic in optim

            loss.backward()
            optimizer.step()
        if n == e_sgds[e_idx] - 1:
            print("save a record of weights for esgd={}".format(e_sgds[e_idx]))

            # save a record for weights
            resulting_weights.append(model.weight.clone())

            e_idx += 1
    return resulting_weights


def continue_train(ho_tr, ho_tt, weights, batch_d, batch_t, t_data, t_targets, t_idx):
  c_tr = torch.cat([batch_d, ho_tr.reshape(add_train_size, 3*32*32)])
  c_tt = torch.cat([batch_t, ho_tt])
  dummy_model = nn.Linear(3*32*32, 2, bias=False)
  with torch.no_grad():
    dummy_model.weight.copy_(weights)
  train_lbfgs(dummy_model, c_tr, c_tt, 1001, 0.00001)
  return (
      test_loss(dummy_model, t_data, t_targets),
      test_accuracy(dummy_model, t_data, t_targets),
      class_accuracy(dummy_model, t_idx)
    )

def continue_sgd_train(
    ho_tr, ho_tt,
    weights,
    batch_d, batch_t,
    t_data, t_targets, t_idx,
    e_max, lr_sgd, e_sgds
):
    c_tr = torch.cat([batch_d, ho_tr.reshape(add_train_size, 3*32*32)])
    c_tt = torch.cat([batch_t, ho_tt])
    dummy_model = nn.Linear(3*32*32, 2, bias=False)
    with torch.no_grad():
        dummy_model.weight.copy_(weights)

    # Obtain weight list for all epoches in e_sgds
    weight_list = train_sgd(dummy_model, c_tr, c_tt, e_max, lr_sgd, weight_decay, e_sgds)
    return weight_list

def c_L_d_theta_regularized(data, target, layerweight):
    """Contained implementation."""
    model = nn.Linear(3*32*32, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(layerweight)
    try:
        layerweight.grad.data.zero_() # zero the gradients
    except:
        pass
    reg = 1e-3 * (layerweight**2).sum() / 2

    loss = criterion(model(data.view(-1, 3*32*32)), target) + reg
    loss.backward()
    return model.weight.grad.data

def c_batch_inf(add_train_data, add_train_target, Lte_H_inv, layerweight):
    Lat = c_L_d_theta_regularized(add_train_data, add_train_target, layerweight)

    try:
      layerweight.grad.data._zero()
    except:
      pass

    return - (Lte_H_inv @ torch.flatten(Lat)).data


def influence(weights, additional_training_sets, batch_idx, batch_d, batch_t, t_data, t_targets):
    """Compute influences"""
    tr_len = len(batch_idx)
    try:
        weights.grad.data.zero_() # zero the gradients
    except:
        pass

    with torch.no_grad():
        weight_clone = weights.clone()

    with timeit("inf compute"):
        with timeit("inf pre-compute"):
            nhes = autograd.functional.hessian(c_grad_curry_regularized(batch_d, batch_t), weights)

            hes_normalized = nhes / tr_len
            Lte = c_L_d_theta_regularized(t_data, t_targets, weight_clone)
            Lte_H_inv = (torch.flatten(Lte) @ torch.cholesky_inverse(torch.reshape(hes_normalized, [3*32*32 * 2, 3*32*32 * 2])))
        infs = [c_batch_inf(add_train_data, add_train_target, Lte_H_inv, weights.clone()) for (add_train_data, add_train_target) in additional_training_sets]

    return infs

def batch_weights_to_test_res(weight_list, t_idx, t_data, t_targets):
    losses = []
    accuracies = []
    class_accuracies = []
    for weight in weight_list:
        # construct a dummy model
        dummy_model = nn.Linear(3*32*32, 2, bias=False)
        with torch.no_grad():
            dummy_model.weight.copy_(weight)
        losses.append(test_loss(dummy_model, t_data, t_targets))
        accuracies.append(test_accuracy(dummy_model, t_data, t_targets))
        class_accuracies.append(class_accuracy(dummy_model, t_idx))
    return (losses, accuracies, class_accuracies)

def ground_truth(weights, additional_training_sets, batch_d, batch_t, t_data, t_targets, t_idx):
    with timeit("ft compute [LBFGS]"):
        finetunes = [continue_train(
                        ho_tr,
                        ho_tt,
                        weights,
                        batch_d,
                        batch_t,
                        t_data,
                        t_targets,
                        t_idx
                    ) for (ho_tr, ho_tt) in additional_training_sets]
    ground_truths = [res[0] for res in finetunes]
    test_accuracies = [res[1] for res in finetunes]
    class_accuracies = [res[2] for res in finetunes]
    return (ground_truths, test_accuracies, class_accuracies)

def limited_fine_tune(
    weights,
    additional_training_sets,
    batch_d,
    batch_t,
    t_data,
    t_targets,
    t_idx,
    e_max,
    lr_sgd,
    e_sgds
):
    with timeit("ft compute limited resource [SGD] e_max={}".format(e_max)):
        limited_fts_weights = [continue_sgd_train(
                            ho_tr,
                            ho_tt,
                            weights,
                            batch_d,
                            batch_t,
                            t_data,
                            t_targets,
                            t_idx,
                            e_max,
                            lr_sgd,
                            e_sgds
                        ) for (ho_tr, ho_tt) in additional_training_sets]

    print("results have components for ", len(limited_fts_weights[0]))
    print("e_sgds={}".format(e_sgds))
    lfts_res = {} # Use a dictionary

    with timeit("ft compute limited resource [SGD-test losses] e_max={}".format(e_max)):
        for e_idx in range(len(e_sgds)):
            e_sgd = e_sgds[e_idx]
            # An intermediate list to add to results
            fixed_ep_weights = [res[e_idx] for res in limited_fts_weights]
            loss, ta, ca = batch_weights_to_test_res(
                fixed_ep_weights,
                t_idx,
                t_data,
                t_targets
            )

            # use the dictionary to store results PER epoch
            lfts_res[e_sgd] = (loss, ta, ca)
    return lfts_res

def run_exp(run, weight_decay=1e-3, add_train_size=440, e_sgds=[1], lr_sgd=0.1):
    # Experiments can be specified in config.yml file, or
    # {
    #     "pretrained_model": nnet,
    #     "training": (batch_idx, batch_d, batch_t),
    #     "testing": (t_idx, t_data, t_targets),
    #     "candidates": unsorted_candidates,
    #     "before_loss"
    # }
    tp = test_prep()
    nnet = tp["pretrained_model"]
    before_loss = tp["before_loss"]
    unsorted_candidates = tp["candidates"]
    additional_training_sets = ([(tr, tt) for (tr, tt, _, _) in unsorted_candidates])

    (batch_idx, batch_d, batch_t) = tp["training"]
    (t_idx, t_data, t_targets) = tp["testing"]

    ## TODO: Group the train and test data
    ground_truths, test_accuracies, class_accuracies = ground_truth(
        nnet.weight.clone(),
        additional_training_sets,
        batch_d,
        batch_t,
        t_data,
        t_targets,
        t_idx
    )
    infs = influence(
        nnet.weight.clone(),
        additional_training_sets,
        batch_idx,
        batch_d,
        batch_t,
        t_data,
        t_targets
    )

    e_max = max(e_sgds)
    lfts_dict = limited_fine_tune(
        nnet.weight.clone(),
        additional_training_sets,
        batch_d,
        batch_t,
        t_data,
        t_targets,
        t_idx,
        e_max,
        lr_sgd,
        e_sgds
    )

    # Some key stats to print
    r1 = [tup[2] for tup in unsorted_candidates]
    r2 = [c_L_d_theta_regularized(dd, tt, nnet.weight.clone()).norm() for (dd,tt) in additional_training_sets]
    r3 = [gt - before_loss for gt in ground_truths]
    r4 = [inf/(add_train_size+len(batch_idx))/10000/len(batch_idx) for inf in infs]

    r5 = test_accuracies
    r6 = [x1 for _, x1 in class_accuracies]

    print("rho(ratio01, gradnorm): ", stats.spearmanr(r1,r2))
    print("rho(ratio01, ground_truths): ", stats.spearmanr(r1,r3))
    print("rho(gradnorm, ground_truths): ", stats.spearmanr(r2, r3))
    print("rho(gradnorm, infs): ", stats.spearmanr(r2, r4))
    print("rho(ratio01, infs): ", stats.spearmanr(r1,r4))
    print("rho(infs, test accuracies): ", stats.spearmanr(r4, r5))
    print("rho(infs, 1-class accuracies): ", stats.spearmanr(r4, r6))

    # Dictionary form storage.
    res = pd.DataFrame()
    res["fraction of candidate set with class `plane`"] = r1
    res["grad_norm"] = [x.item() for x in r2]
    res["improved_loss"] = [x.item() for x in r3]
    res["after_loss"] = [x.item() for x in ground_truths]
    res["influence"] = [x.item() for x in infs]
    res["adjusted_influence"] = [x.item() for x in r4]
    res["test_accuracies"] = r5
    res["class_0_acc"] = [x0 for x0, _ in class_accuracies]
    res["precision"] = [x1 for _, x1 in class_accuracies]
    res["price"] = [max(0, -aix) for aix in res["adjusted_influence"]]

    for e_sgd in e_sgds:
        res["ft_loss_e{}lr{}".format(e_sgd, lr_sgd)] = lfts_dict[e_sgd][0]

    # there is yet still noise level, and lr sensitivity
    filename = "exp-imbalance-res-100-eps{}-lr{}-run{}".format(e_sgds, lr_sgd, run)
    print("saving to file {}.csv".format(filename))
    res.to_csv(filename)
    # One last test
    r8s = [[(after_loss - before_loss) for after_loss in lfts_dict[e_sgd][0]] for e_sgd in e_sgds]

def main(argv):
    del argv  # Unused.
    ## Parse arguments
    weight_decay = FLAGS.weight_decay
    add_train_size = FLAGS.add_train_size
    runs = FLAGS.runs
    e_sgds_strs = FLAGS.e_sgds
    lr_sgd = FLAGS.lr_sgd

    for run in range(runs):
        print("--- start of run{}".format(run))
        run_exp(run, e_sgds=[int(x) for x in e_sgds_strs], lr_sgd=lr_sgd)

if __name__ == '__main__':
  app.run(main)