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
        self.num_negative_samples = num_negative_samples
        self.is_embedding_sparse = nn_embedding_kwargs.get("sparse", None)
        self.learning_rate = learning_rate

        # negative noise sampling using weights
        if weights is None:
            # sample uniformly
            weights = torch.ones(self.num_embeddings)

        # Register negative sample weights torch Tensor within this nn module
        # Required for PyTorch to put on correct device for sampling
        self.register_buffer("negative_sample_weights", weights)

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

    def forward(self, anchors, targets):
        # Fetch batch size
        batch_size = anchors.size(0)

        # embedding lookups
        anchors_embeddings = self.embeddings(anchors)
        target_embeddings = self.target_embeddings(targets)

        # sample negative noise items from the distribution
        # TODO make sure noise samples are not part of context
        negatives = torch.multinomial(
            self.negative_sample_weights, batch_size * self.num_negative_samples, replacement=True
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
