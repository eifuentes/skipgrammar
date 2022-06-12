"""
Skip-gram Negative Sampling (SGNS) Neural Network.
"""
import torch
import torch.nn as nn


def negative_sampling_loss(atensor, ctensor, ntensor):
    # TODO: check input tensor shapes, raise error otherwise
    batch_size, dim = atensor.size()
    num_negative_samples = ntensor.size(0) // batch_size

    if batch_size == 1:
        raise RuntimeError("negative_sampling_loss assumes batch_size > 1")

    # reshape tensors
    atensor = atensor.view(batch_size, dim, 1)  # batch of column vectors
    ctensor = ctensor.view(batch_size, 1, dim)  # batch of row vectors
    ntensor = ntensor.view(
        batch_size, num_negative_samples, dim
    )  # batch of row vectors for each negative sample

    # calcuate loss
    loss = torch.bmm(ctensor, atensor).sigmoid().log().squeeze()
    noise_loss = torch.bmm(ntensor.neg(), atensor).sigmoid().log().squeeze()
    noise_loss = noise_loss.sum(1)  # sum over drawn noise samples
    return -(loss + noise_loss).mean()  # aggregate loss across batch size


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, atensor, ctensor, ntensor):
        return negative_sampling_loss(atensor, ctensor, ntensor)


class SGNS(nn.Module):
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
    ):
        super().__init__()
        self.num_embeddings_ = int(num_embeddings)
        self.embedding_dim_ = int(embedding_dim)
        self.weights_ = weights
        self.num_negative_samples_ = num_negative_samples

        # embeddings lookups
        self.embeddings = nn.Embedding(
            num_embeddings=self.num_embeddings_,
            embedding_dim=self.embedding_dim_,
            **nn_embedding_kwargs
        )
        self.target_embeddings = nn.Embedding(
            num_embeddings=self.num_embeddings_,
            embedding_dim=self.embedding_dim_,
            **nn_embedding_kwargs
        )

        # initilize parameters
        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.5 / self.embedding_dim_
        nn.init.uniform_(self.embeddings.weight, -init_range, init_range)
        nn.init.constant_(self.target_embeddings.weight, 0.0)

    def forward(self, anchors, target):
        # infer batch size dynamically
        batch_size = anchors.size(0)

        # embedding lookups
        anchors_embeddings = self.embeddings(anchors)
        target_embeddings = self.target_embeddings(target)

        # negative noise sampling using weights
        if self.weights_ is None:
            # sample uniformly
            weights = torch.ones(self.num_embeddings_)
        else:
            # sample via provided weights
            weights = self.weights_

        # sample negative noise items from the distribution
        # TODO make sure noise samples are not part of context
        negatives = torch.multinomial(
            weights, batch_size * self.num_negative_samples_, replacement=True
        )
        negative_embeddings = self.target_embeddings(negatives)

        return anchors_embeddings, target_embeddings, negative_embeddings

    def as_embedding(self, index):
        return self.embeddings(torch.tensor(index))
