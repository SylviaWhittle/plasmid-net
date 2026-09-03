"""Scripts for training the model."""

import random
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import hessian
from skimage.morphology import skeletonize

from src.resunet import ResUNet
from src.loss import BCEWithLogitsDiceLoss

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "resunet_model.pth"
BASE_DIR = Path("/Users/sylvi/topo_data/dna-damage-unet")
PREDICTIONS_DIR = BASE_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)  # create the directory if it doesn't exist

SAMPLE_TYPES = ["cesium"]

NUM_SAMPLES_PER_TYPE = [1000]
SEED = 0
TRAINING_DATA_DIR = BASE_DIR / "data"
ALLOW_TOO_FEW_SAMPLES = True
AUGMENTATION_FLIP_ROT = True
AUGMENTATION_SCALE = False
AUGMENTATION_SCALE_MAX_ZOOM_PERCENTAGE = 0.2

EXTRA_CHANNELS_HESSIAN = False  # whether to add hessian channel to the input images
EXTRA_CHANNELS_HESSIAN_SIGMAS = [3.0]  # list of sigmas to use for the hessian filter
EXTRA_CHANNELS_HESSIAN_NORMALISED = True

RESIZE_TO_SIZE = 256  # resize images and masks to this size for training and validation
# normalisation values for the images
VMIN = -1.0
VMAX = 5.0

IN_CHANNELS = 1 + len(EXTRA_CHANNELS_HESSIAN_SIGMAS) if EXTRA_CHANNELS_HESSIAN else 1
OUT_CHANNELS = 1

# eval
EVAL_HEIGHT_THRESHOLD = (1.1 - VMIN) / (VMAX - VMIN)  # threshold for classical height-based thresholding


def seed_everything(seed: int) -> None:
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def sample_type_split(
    sample_type_dirs: list[Path],
    num_samples_per_type: list[int],
    allow_too_few_samples: bool = False,
) -> tuple[list[Path], list[Path], dict[str, int]]:
    """
    Grab images from each sample type for the dataset.

    Parameters
    ----------
    sample_type_dirs : list[Path]
        List of directories containing the sample types.
    num_samples_per_type : list[int]
        List of the number of samples to grab from each sample type.
    allow_too_few_samples : bool, optional
        Whether to allow fewer samples than requested if a sample type has too few samples, by default False.

    Returns
    -------
    tuple[list[Path], list[Path], dict[str, int]]
        A tuple containing two lists: the first list contains the paths to the sampled image files,
        and the second list contains the paths to the corresponding mask files.
        The third element is a dictionary with the number of samples actually used per sample type.
    """
    assert len(sample_type_dirs) == len(
        num_samples_per_type
    ), "sample_type_dirs and num_samples_per_type must have the same length"

    sampled_image_files = []
    sampled_mask_files = []
    num_samples_used = {}

    print(f"Sampling {num_samples_per_type} files from each sample type directory in {sample_type_dirs}.")

    for sample_type_dir, num_samples in zip(sample_type_dirs, num_samples_per_type):
        # get all the images and masks in the sample type directory
        image_files = sorted(list((sample_type_dir).glob("**/group_*_ready/*image.npy")))
        mask_files = sorted(list((sample_type_dir).glob("**/group_*_ready/*label.npy")))

        print(f"| Found {len(image_files)} images and {len(mask_files)} masks in {sample_type_dir}.")
        assert len(image_files) == len(mask_files), f"Number of images and masks must be the same in {sample_type_dir}"

        if len(image_files) < num_samples:
            if allow_too_few_samples:
                num_samples = len(image_files)
            else:
                raise ValueError(
                    f"| Not enough samples in {sample_type_dir}. Requested {num_samples}, but only {len(image_files)} available."
                )
        # sample the files
        sampled_indexes = torch.randperm(len(image_files))[:num_samples].tolist()
        sampled_image_files.extend([image_files[i] for i in sampled_indexes])
        sampled_mask_files.extend([mask_files[i] for i in sampled_indexes])
        num_samples_used[sample_type_dir.name] = num_samples

    return sampled_image_files, sampled_mask_files, num_samples_used


def train_val_split(
    images: list[Path],
    masks: list[Path],
    val_split: float,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """
    Split the training data into training and validation sets.

    Parameters
    ----------
    images : list[Path]
        List of paths to the image files.
    masks : list[Path]
        List of paths to the mask files.
    val_split : float
        Fraction of the data to use for validation.
    """

    assert len(images) == len(masks), "Number of images and masks must be the same"

    # shuffle the data
    indices = torch.randperm(len(images))
    images = [images[i] for i in indices]
    masks = [masks[i] for i in indices]

    # split the data into training and validation sets
    val_size = int(len(images) * val_split)
    val_images = images[:val_size]
    val_masks = masks[:val_size]
    train_images = images[val_size:]
    train_masks = masks[val_size:]

    return train_images, train_masks, val_images, val_masks


def get_loaders(
    images_paths: list[Path],
    masks_paths: list[Path],
    val_split: float,
    batch_size: int,
    vmin: float,
    vmax: float,
    resize_to_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Get the training and validation data loaders.

    Parameters
    ----------
    images_paths : list[Path]
        List of paths to the image files.
    masks_paths : list[Path]
        List of paths to the mask files.
    val_split : float
        Fraction of the data to use for validation.
    batch_size : int
        Batch size for the data loaders.
    vmin : float
        Minimum value for normalising the images.
    vmax : float
        Maximum value for normalising the images.
    resize_to_size : int | None, optional
        Resize images and masks to this size, by default None (no resizing).

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation data loaders.
    """
    train_image_files, train_mask_files, val_image_files, val_mask_files = train_val_split(
        images_paths, masks_paths, val_split
    )

    train_dataset = SegmentationDataset(
        image_files=train_image_files,
        mask_files=train_mask_files,
        vmin=vmin,
        vmax=vmax,
        augment_flip_rot=AUGMENTATION_FLIP_ROT,
        augment_scale=AUGMENTATION_SCALE,
        augment_max_zoom_percentage=AUGMENTATION_SCALE_MAX_ZOOM_PERCENTAGE,
        resize_to_size=resize_to_size,
    )
    val_dataset = SegmentationDataset(
        image_files=val_image_files,
        mask_files=val_mask_files,
        vmin=vmin,
        vmax=vmax,
        augment_flip_rot=False,
        augment_scale=False,
        augment_max_zoom_percentage=AUGMENTATION_SCALE_MAX_ZOOM_PERCENTAGE,
        resize_to_size=resize_to_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
):
    # train the model for one epoch
    model.train()  # this sets the model to training mode
    # keep track of the running loss for this epoch
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device)

        # zero the exising gradients - so that they don't accumulate
        optimiser.zero_grad()

        # forward pass
        outputs = model(images)

        # compute the loss
        loss = criterion(outputs, targets)

        # backward pass & optimisation
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)  # multiply by batch size to get total loss for this batch
        progress_bar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)  # average loss for this epoch
    return epoch_loss


@torch.no_grad()  # disable gradient calculation for validation
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()  # set the model to evaluation mode
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)  # multiply by batch size to get total loss for this batch

    epoch_loss = running_loss / len(dataloader.dataset)  # average loss for this epoch
    return epoch_loss


@torch.no_grad()  # disable gradient calculation for validation
def validate_dice(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    confidence_threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    model.eval()
    dice_total = 0.0
    num_images = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        probabilities = torch.sigmoid(model(images))  # convert logits to probabilities
        predictions = (
            probabilities > confidence_threshold
        ).float()  # threshold the probabilities to get binary predictions

        # compute dice
        predictions = predictions.flatten(1)  # flatten the tensor - ie from [B, C, H, W] to [B, C*H*W]
        targets = targets.flatten(1)

        intersection = (predictions * targets).sum(1)
        union = predictions.sum(1) + targets.sum(1)
        dice = (2.0 * intersection + eps) / (union + eps)  # dice score for each image in the batch
        dice_total += dice.sum().item()  # .item() is to convert the tensor to a python float
        num_images += images.size(0)

    return 1.0 - (dice_total / num_images)  # average dice score over the dataset


def zoom_and_shift(
    image: torch.Tensor, ground_truth: torch.Tensor, max_zoom_percentage: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scale and translate image and corresponding ground truth mask.

    Zooms in on the image/mask by a random amount between 0 and
    max_zoom_percentage, then shifts the image/mask by a random amount
    up to the number of zoomed pixels.

    Parameters
    ----------
    image : torch.Tensor
        The input image tensor of shape [C, H, W].
    ground_truth : torch.Tensor
        The corresponding ground truth mask tensor of shape [C, H, W].
    max_zoom_percentage : float, optional
        The maximum percentage of the image to zoom in, by default 0.1.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The zoomed and shifted image and ground truth mask tensors. [C, H, W]
    """
    # check sizing
    assert image.ndim == 3, f"Image must be 3D tensor [C, H, W], got {image.ndim}D tensor"
    assert ground_truth.ndim == 3, f"Ground truth must be 3D tensor [C, H, W], got {ground_truth.ndim}D tensor"
    assert (
        image.shape[1:] == ground_truth.shape[1:]
    ), f"Image and ground truth must have the same height and width, got {image.shape[1:]} and {ground_truth.shape[1:]}"
    original_size = image.shape[-2:]  # [H, W]
    # Choose a zoom percentage and calculate the number of pixels to zoom in
    zoom = np.random.uniform(0, max_zoom_percentage)
    zoom_pixels = int(zoom * image.shape[1])

    # If there is zoom, choose a random shift
    if int(zoom_pixels) > 0:
        shift_x = np.random.randint(int(-zoom_pixels), int(zoom_pixels))
        shift_y = np.random.randint(int(-zoom_pixels), int(zoom_pixels))

        # Zoom and shift  the image and ground truth mask
        image = image[:, zoom_pixels + shift_y : -zoom_pixels + shift_y, zoom_pixels + shift_x : -zoom_pixels + shift_x]
        ground_truth = ground_truth[
            :, zoom_pixels + shift_y : -zoom_pixels + shift_y, zoom_pixels + shift_x : -zoom_pixels + shift_x
        ]

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),  # add a batch dimension for interpolation
            size=original_size,
            mode="bilinear",
            align_corners=False,  # apparently this is default for bilinear
        ).squeeze(
            0
        )  # remove the batch dimension

        ground_truth = torch.nn.functional.interpolate(
            ground_truth.unsqueeze(0),  # add a batch dimension for interpolation
            size=original_size,
            mode="nearest",  # use nearest neighbour for masks to avoid interpolation artifacts
        ).squeeze(
            0
        )  # remove the batch dimension

    return image, ground_truth


class SegmentationDataset(torch.utils.data.Dataset):
    """Custom dataset for augmenting the segmentation data."""

    def __init__(
        self,
        image_files: list[Path],
        mask_files: list[Path],
        vmin: float,
        vmax: float,
        augment_flip_rot: bool,
        augment_scale: bool,
        augment_max_zoom_percentage: float,
        resize_to_size: int | None,
    ) -> None:
        """Initialise."""
        self.image_files = image_files
        self.mask_files = mask_files
        self.vmin = vmin
        self.vmax = vmax
        self.augment_flip_rot = augment_flip_rot
        self.augment_scale = augment_scale
        self.augment_max_zoom_percentage = augment_max_zoom_percentage
        self.resize_to_size = resize_to_size

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset and augment if needed."""
        image_original = torch.from_numpy(np.load(self.image_files[index])).float()  # [H, W]
        mask = torch.from_numpy(np.load(self.mask_files[index]).astype(bool)).float()  # [C, H, W] | [H, W]

        if self.resize_to_size is not None:
            # resize image if needed
            image_original = (
                torch.nn.functional.interpolate(
                    image_original.unsqueeze(0).unsqueeze(0),  # add batch and channel dimensions for interpolation
                    size=(self.resize_to_size, self.resize_to_size),
                    mode="bilinear",
                    align_corners=False,  # apparently this is default for bilinear
                )
                .squeeze(0)
                .squeeze(0)
            )  # remove the batch and channel dimensions
            # resize mask if needed
            mask = (
                torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),  # add batch and channel dimensions for interpolation
                    size=(self.resize_to_size, self.resize_to_size),
                    mode="nearest",  # use nearest neighbour for masks to avoid interpolation artifacts
                )
                .squeeze(0)
                .squeeze(0)
            )  # remove the batch and channel dimensions
            # ensure the mask is still binary after resizing
            mask = (mask > 0.5).float()

        # add channel dim if missing
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [C, H, W]

        # normalise the image
        image = image_original.clone()
        image = torch.clamp(image, self.vmin, self.vmax)
        image = (image - self.vmin) / (self.vmax - self.vmin)
        # add channel dimension to the image tensor
        image = image.unsqueeze(0)  # [C, H, W]

        if EXTRA_CHANNELS_HESSIAN:
            # Add hessian channel to the image tensor for better feature extraction
            # Calculate the hessian before normalising the image
            image_hessian = hessian(
                image=image_original.squeeze(0).numpy(),
                sigmas=EXTRA_CHANNELS_HESSIAN_SIGMAS,
                mode="reflect",
                # beta = 0.1,
                # scale_step = 1.0,
                # scale_range = (1, 10),
            )

            # put it into the image tensor
            image_hessian = torch.from_numpy(image_hessian).float().unsqueeze(0)  # [C, H, W]
            # normalise the hessian channel if needed
            if EXTRA_CHANNELS_HESSIAN_NORMALISED:
                image_hessian = torch.clamp(image_hessian, self.vmin, self.vmax)
                image_hessian = (image_hessian - self.vmin) / (self.vmax - self.vmin)
            # concatenate the hessian channel to the image tensor
            image = torch.cat((image, image_hessian), dim=0)  # [C, H, W]

        if self.augment_flip_rot:
            # horizontal flip
            if torch.rand(1).item() < 0.5:
                image = image.flip(-1)
                mask = mask.flip(-1)
            # vertical flip
            if torch.rand(1).item() < 0.5:
                image = image.flip(-2)
                mask = mask.flip(-2)
            rotation = int(torch.randint(0, 4, (1,)).item())
            image = torch.rot90(image, rotation, [-2, -1])
            mask = torch.rot90(mask, rotation, [-2, -1])
        if self.augment_scale:
            image, mask = zoom_and_shift(
                image=image,
                ground_truth=mask,
                max_zoom_percentage=self.augment_max_zoom_percentage,
            )

        return image, mask


def main():

    seed_everything(SEED)

    # Grab samples from directories for each sample type
    train_image_files, train_mask_files, num_samples_used = sample_type_split(
        sample_type_dirs=[TRAINING_DATA_DIR / sample_type for sample_type in SAMPLE_TYPES],
        num_samples_per_type=NUM_SAMPLES_PER_TYPE,
        allow_too_few_samples=ALLOW_TOO_FEW_SAMPLES,
    )

    # get data loaders
    train_loader, val_loader = get_loaders(
        images_paths=train_image_files,
        masks_paths=train_mask_files,
        val_split=0.2,
        batch_size=BATCH_SIZE,
        vmin=VMIN,
        vmax=VMAX,
        resize_to_size=RESIZE_TO_SIZE,
    )

    # initialise model, loss function, and optimiser
    model = ResUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    # criterion = BCEWithLogitsDiceLoss(bce_weight=0.5)
    # add positive weighting
    positive_pixels = 0
    negative_pixels = 0
    for mask_path in train_mask_files:
        mask = np.load(mask_path).astype(bool)
        positive_pixels += np.sum(mask)
        negative_pixels += np.sum(~mask)
    positive_weight = negative_pixels / max(positive_pixels, 1)
    weighted_criterion = BCEWithLogitsDiceLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=DEVICE)
    )
    unweighted_criterion = BCEWithLogitsDiceLoss(pos_weight=None)
    # use adam since using batchnorm and relu
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Gradually drop the learning rate if the validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")

    print("\n--- Starting training ---\n")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, weighted_criterion, optimiser, DEVICE)
        weighted_val_loss = validate(model, val_loader, weighted_criterion, DEVICE)
        unweighted_val_loss = validate(model, val_loader, unweighted_criterion, DEVICE)
        val_dice_score = validate_dice(model, val_loader, DEVICE, confidence_threshold=0.5)

        # step the LR scheduler with the validation loss
        scheduler.step(weighted_val_loss)
        current_lr = optimiser.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Weighted Loss: {weighted_val_loss:.4f}, Unweighted Loss: {unweighted_val_loss:.4f}, Dice Score: {val_dice_score:.4f}, LR: {current_lr:.6f}"
        )

        # save checkpoint if it's the best so far
        if val_dice_score < best_val_loss:
            best_val_loss = val_dice_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")

    print("\n--- Training complete ---\n")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Training stats:")
    print(f"  - Device: {DEVICE}")
    print(f"  - Seed: {SEED}")
    print(f"  - Total epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Initial learning rate: {LEARNING_RATE}")
    print(f"  - Model saved to: {MODEL_SAVE_PATH}")
    print(f"  - Number of samples requested per type: {NUM_SAMPLES_PER_TYPE}")
    print(f"  - Number of samples actually used per type: {num_samples_used}")
    print(f"  - Sample types: {SAMPLE_TYPES}")
    print("  - Extra channels")
    print(f"    - Hessian : {EXTRA_CHANNELS_HESSIAN}")
    print(f"      - Hessian normalised : {EXTRA_CHANNELS_HESSIAN_NORMALISED}")
    print(f"      - Hessian sigmas : {EXTRA_CHANNELS_HESSIAN_SIGMAS}")
    print("  - Augmentation:")
    print(f"    - Flip and rotate: {AUGMENTATION_FLIP_ROT}")
    print(f"    - Scale: {AUGMENTATION_SCALE}")

    # Load the best model and evaluate on the validation set
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    weighted_val_loss = validate(model, val_loader, weighted_criterion, DEVICE)
    print(f"Best model validation loss: {weighted_val_loss:.4f}")
    val_dice_score = validate_dice(model, val_loader, DEVICE, confidence_threshold=0.5)
    print(f"Best model validation dice score: {val_dice_score:.4f}")
    # Plot some predictions from the validation set
    model.eval()
    with torch.no_grad():
        num_rows = len(val_loader)
        num_cols = IN_CHANNELS + OUT_CHANNELS * 2 + 1 + 1 + 1 + 1
        # input channels
        # target channels
        # classical threshold binary
        # classical threshold skeleton
        # predicted channels
        # thresholded predicted
        # skeletonized thresholded predicted
        plt.figure(figsize=(15, 5 * num_rows))
        for i, (images, targets) in enumerate(val_loader):
            col_index = 0
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # convert logits to probabilities since logits are in the
            # range (-inf, inf) and we want probabilities in the range (0, 1)

            # plot the image

            plt.subplot(num_rows, num_cols, i * num_cols + col_index + 1)
            plt.imshow(images[0, 0].cpu(), cmap="gray")
            plt.title("Input Image")
            plt.axis("off")
            col_index += 1

            if EXTRA_CHANNELS_HESSIAN:
                # plot the hessian channel
                plt.subplot(num_rows, num_cols, i * num_cols + col_index + 1)
                plt.imshow(images[0, 1].cpu(), cmap="gray")
                plt.title("Hessian Channel")
                plt.axis("off")
                col_index += 1

            # plot the target mask channels
            plt.subplot(num_rows, num_cols, i * num_cols + col_index + 1)
            plt.imshow(targets[0, 0].cpu(), cmap="gray")
            plt.title("Target Mask")
            plt.axis("off")
            col_index += 1

            # classical threshold binary mask
            classical_threshold_mask = (images[0, 0] > EVAL_HEIGHT_THRESHOLD).float()
            plt.subplot(num_rows, num_cols, i * num_cols + col_index + 1)
            plt.imshow(classical_threshold_mask.cpu(), cmap="gray")
            plt.title("Threshold Binary Mask")
            plt.axis("off")
            col_index += 1

            classical_threshold_mask_skeleton = skeletonize(classical_threshold_mask.cpu().numpy())
            plt.subplot(num_rows, num_cols, i * num_cols + col_index + 1)
            plt.imshow(classical_threshold_mask_skeleton, cmap="gray")
            plt.title("Threshold Skeletonized Mask")
            plt.axis("off")
            col_index += 1

            # plot the predicted mask channels
            plt.subplot(num_rows, num_cols, i * num_cols + col_index + 1)
            plt.imshow(outputs[0, 0].cpu(), cmap="gray")
            plt.title("Predicted Mask")
            plt.axis("off")
            col_index += 1

            # plot the thresholded predicted mask
            thresholded_outputs = (outputs[0, 0] > 0.5).float()
            plt.subplot(num_rows, num_cols, i * num_cols + col_index + 1)
            plt.imshow(thresholded_outputs.cpu(), cmap="gray")
            plt.title("Thresholded Predicted Mask")
            plt.axis("off")
            col_index += 1

            # plot the skeletonized thresholded predicted mask
            skeletonized_outputs = skeletonize(thresholded_outputs.cpu().numpy())
            plt.subplot(num_rows, num_cols, i * num_cols + col_index + 1)
            plt.imshow(skeletonized_outputs, cmap="gray")
            plt.title("Skeletonized Predicted Mask")
            plt.axis("off")
            col_index += 1

        plt.tight_layout()
        # save the figure
        # create a guid for the image based on index and epoch
        image_guid = f"val_predictions_epoch_{EPOCHS}"
        plt.savefig(PREDICTIONS_DIR / f"prediction_{image_guid}.png")


if __name__ == "__main__":
    main()
