from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
import torch


def data_loader(train_inputs, test_inputs, train_labels, test_labels, batch_size=50):

    train_inputs, test_inputs, train_labels, test_labels = \
        tuple(torch.tensor(data) for data in [train_inputs, test_inputs, train_labels, test_labels])

    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader
