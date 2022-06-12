import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from skipgrammar.models.sgns import negative_sampling_loss


class SGNS(LightningModule):
    """
    >>> SGNS(1000, nn_embedding_kwargs={'sparse': True})
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim=300,
        weights=None,
        num_negative_samples=5,
        nn_embedding_kwargs=dict(),
        learning_rate=0.003,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weights = weights
        self.num_negative_samples = num_negative_samples
        self.is_embedding_sparse = nn_embedding_kwargs.get("sparse", None)
        self.learning_rate = learning_rate

        # embeddings lookups
        self.embeddings = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            **nn_embedding_kwargs
        )
        self.target_embeddings = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            **nn_embedding_kwargs
        )

        # initilize parameters
        self.reset_parameters()

        # save hyperparameters
        self.save_hyperparameters()

    def reset_parameters(self):
        init_range = 0.5 / self.embedding_dim
        nn.init.uniform_(self.embeddings.weight, -init_range, init_range)
        nn.init.constant_(self.target_embeddings.weight, 0.0)

    def forward(self, anchors, target):
        # infer batch size dynamically
        batch_size = anchors.size(0)

        # embedding lookups
        anchors_embeddings = self.embeddings(anchors)
        target_embeddings = self.target_embeddings(target)

        # negative noise sampling using weights
        if self.weights is None:
            # sample uniformly
            weights = torch.ones(self.num_embeddings)
        else:
            # sample via provided weights
            if isinstance(self.weights, pd.Series):
                weights = torch.Tensor(self.weights.values)
            elif isinstance(self.weights, np.array):
                weights = torch.Tensor(self.weights)
            else:
                weights = self.weights

        # sample negative noise items from the distribution
        # TODO make sure noise samples are not part of context
        negatives = torch.multinomial(
            weights, batch_size * self.num_negative_samples, replacement=True
        )
        negative_embeddings = self.target_embeddings(negatives)

        return anchors_embeddings, target_embeddings, negative_embeddings

    def as_embedding(self, index):
        return self.embeddings(torch.tensor(index))

    def training_step(self, batch, batch_idx):
        anchors, targets = batch
        anchors_embeddings, target_embeddings, negative_embeddings = self(
            anchors, targets
        )
        loss = negative_sampling_loss(
            anchors_embeddings, target_embeddings, negative_embeddings
        )
        tensorboard_logs = {"training_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return (
            torch.optim.SparseAdam(self.parameters(), lr=self.learning_rate)
            if self.is_embedding_sparse
            else torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        )
