"""Loss functions for training and evaluation."""

import torch
import torch.nn as nn


class PermutationInvariantDiceLoss_2_channel(nn.Module):
    """Permutation invariant dice loss for agnostic multi class segmentation loss."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def _dice_loss(self, y_true, y_pred):
        """Compute the dice loss for a single class."""
        # note: channel indexes for these tensors are [batch, channel, height, width]
        # hence summing over only the height and width dimensions (2, 3)
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

    def forward(self, logits, targets):
        """
        Compute the permutation invariant dice loss for 2 channels.

        Parameters
        ----------
        logits : torch.Tensor
            The predicted logits
        targets : torch.Tensor
            The ground truth labels

        Returns
        -------
        torch.Tensor
            The computed loss.
        """

        # apply independent sigmoids since channels can overlap
        # explanation: if there are 2 channels, the model can predict both channels as 1 for a given pixel, which
        # is valid, so we need to apply independent sigmoids to each channel to get the predicted
        # probabilities for each channel.
        # a sigmoid is applied because the model outputs logits, and we want to convert them to probabilities in
        # the range [0, 1] for each channel independently. Logits can be any real number.
        y_pred = torch.sigmoid(logits)

        # slice out the predictions and targets for each channel.
        # note that we need to slice and not index, so that it forces the output to have the same number of
        # dimensions as the input which is required for dice loss computation.
        # slice out predictions
        y_pred_1 = y_pred[:, 0:1, :, :]
        y_pred_2 = y_pred[:, 1:2, :, :]

        # slice out targets
        y_true_1 = targets[:, 0:1, :, :]
        y_true_2 = targets[:, 1:2, :, :]

        # print(f"y_pred_1 shape: {y_pred_1.shape}, y_pred_2 shape: {y_pred_2.shape}")
        # print(f"y_true_1 shape: {y_true_1.shape}, y_true_2 shape: {y_true_2.shape}")

        # print(targets.min().item(), targets.max().item())
        # print(torch.unique(targets))

        # compute dice loss for both permutations
        loss_1 = (self._dice_loss(y_true_1, y_pred_1) + self._dice_loss(y_true_2, y_pred_2)) / 2
        loss_2 = (self._dice_loss(y_true_1, y_pred_2) + self._dice_loss(y_true_2, y_pred_1)) / 2
        # return the minimum loss
        loss = torch.min(loss_1, loss_2)
        # need to return the mean loss over the batch, since the dice loss is computed per sample in the batch
        return loss.mean()


def dice_loss(logits, target, eps=1e-6):
    """Compute the Dice loss between predicted and target tensors."""

    # pred = pred.sigmoid() if pred.dtype.is_floating_point and pred.max() > 1 else pred # - this assumes probabilities
    # if the maximum logit is at most 1.

    pred = torch.sigmoid(logits)
    # pred_flat = pred.view(pred.size(0), -1)
    pred_flat = pred.flatten(1)  # flatten the tensor - ie from [B, C, H, W] to [B, C*H*W]
    # targ_flat = target.view(target.size(0), -1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class BCEWithLogitsDiceLoss(nn.Module):
    """Combined BCE with logits and Dice loss."""

    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        d_loss = dice_loss(logits, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * d_loss
