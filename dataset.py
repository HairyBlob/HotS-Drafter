# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:23:04 2017

@author: Daniel
"""
import numpy as np
#Dataset class adapted from the mnist example. Mostly useful for its minibatch method
class DataSet(object):

    def __init__(self,
           teams,
           maps,
           labels):
        assert teams.shape[0] == labels.shape[0], (
        'teams.shape: %s labels.shape: %s' % (teams.shape, labels.shape))
        self._num_examples = teams.shape[0]
        self._teams = teams
        self._maps = maps
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def teams(self):
        return self._teams
    
    @property
    def maps(self):
        return self._maps

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffling=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffling:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._teams = self.teams[perm0]
            self._maps = self.maps[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            teams_rest_part = self._teams[start:self._num_examples]
            maps_rest_part = self._maps[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffling:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._teams = self.teams[perm]
                self._maps = self.maps[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            teams_new_part = self._teams[start:end]
            maps_new_part = self._maps[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((teams_rest_part, teams_new_part), axis=0) , np.concatenate((maps_rest_part, maps_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._teams[start:end], self._maps[start:end], self._labels[start:end]

class DataSet_discriminator(object):

    def __init__(self,
           teams,
           labels):
        assert teams.shape[0] == labels.shape[0], (
        'teams.shape: %s labels.shape: %s' % (teams.shape, labels.shape))
        self._num_examples = teams.shape[0]
        self._teams = teams
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def teams(self):
        return self._teams

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffling=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffling:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._teams = self.teams[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            teams_rest_part = self._teams[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffling:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._teams = self.teams[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            teams_new_part = self._teams[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((teams_rest_part, teams_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._teams[start:end], self._labels[start:end]