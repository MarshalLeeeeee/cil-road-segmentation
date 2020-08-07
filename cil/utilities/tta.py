import numpy as np
import torch

# If the predicted probability is higher than the following value, it is considered a vote
vote_prob_threshold = 0.25
# If the ratio of positive voter is high thant the following value, positive result is accepted
vote_acc_threshold = 0.25

original_shape = (608, 608)
crop_shape = (400, 400)

crop_slices = [
    np.s_[:, :, :crop_shape[0], :crop_shape[1]],
    np.s_[:, :, :crop_shape[0], -crop_shape[1]:],
    np.s_[:, :, -crop_shape[0]:, :crop_shape[1]],
    np.s_[:, :, -crop_shape[0]:, -crop_shape[1]:],
]

crop_count = None

flips = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.flip(x, dims=[3]),
]

rotations = [
    lambda x: x,
    lambda x: torch.rot90(x, 1, dims=[2, 3]),
    lambda x: torch.rot90(x, 2, dims=[2, 3]),
    lambda x: torch.rot90(x, 3, dims=[2, 3])
]

flips_reverse = flips

rotations_reverse = [
    lambda x: x,
    lambda x: torch.rot90(x, 3, dims=[2, 3]),
    lambda x: torch.rot90(x, 2, dims=[2, 3]),
    lambda x: torch.rot90(x, 1, dims=[2, 3])
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TTA:
    """
    Test-time Augmentation Utilities.
    """

    @staticmethod
    def transform_batch(image_batch):
        """
        Apply lossless transformation.
        input: a 4D tensor.
        output: a list of 4D tensors. Each tensors have been applied same transformation.
        """
        batches = list()
        for flip in flips:
            for rotation in rotations:
                batches.append(flip(rotation(image_batch)))

        return batches

    @staticmethod
    def reverse_transform(image_batches, strategy='avg'):
        """
        Undo transformation.
        input: a list of 4D tensors.
        output: a single 4D tensor, aggregating over all reverse-transformed tensors.
        """
        result_batch = torch.zeros_like(image_batches[0], device=device)

        count = 0
        for flip in flips_reverse:
            for rotation in rotations_reverse:
                # In the reverse phase we need to undo rotation first
                batch = rotation(flip(image_batches[count]))
                if strategy == 'avg':
                    result_batch += batch
                elif strategy == 'vote':
                    result_batch += (batch >= vote_prob_threshold)
                elif strategy == 'max':
                    result_batch = torch.max(result_batch, batch)
                count += 1

        if strategy == 'avg':
            return result_batch / float(count)
        elif strategy == 'vote':
            return ((result_batch / float(count)) > vote_acc_threshold).float()
        elif strategy == 'max':
            return result_batch

    @staticmethod
    def crop_batch(image_batch):
        """
        Crop images in predefined ways (specified by slice objects `crop_slices`).
        input: a 4D tensor.
        output: a list of 4D tensors. Each tensors have been applied same transformation.
        """
        return [image_batch[crop_slice] for crop_slice in crop_slices]

    @staticmethod
    def reverse_crop(cropped_batches, strategy='avg'):
        """
        Crop images in predefined ways (specified by slice objects `crop_slices`).
        input: a list of 4D tensors.
        output: a single 4D tensor, aggregating cropped results by strategy.
        """
        global crop_count
        if crop_count is None:
            crop_count = torch.zeros((1, 1, *original_shape), device=device)
            for crop_slice in crop_slices:
                crop_count[crop_slice] += 1.

        batch_shape = (*cropped_batches[0].shape[:2], *original_shape)
        result_batch = torch.zeros(batch_shape, device=device)

        for cropped_batch, crop_slice in zip(cropped_batches, crop_slices):
            if strategy == 'avg':
                result_batch[crop_slice] += cropped_batch
            elif strategy == 'vote':
                result_batch[crop_slice] += (cropped_batch >= vote_prob_threshold)
            elif strategy == 'max':
                result_batch[crop_slice] = torch.max(result_batch[crop_slice], cropped_batch)

        if strategy == 'avg':
            return result_batch / crop_count
        elif strategy == 'vote':
            return ((result_batch / crop_count) > vote_acc_threshold).float()
        elif strategy == 'max':
            return result_batch

    @staticmethod
    def augmented_predict(predict, tensor, strategy='avg', tta_crop=True):
        """
        Predict with test-time augmentation:
        - lossless transformation: flip and rotation (90 degree)
        - cropping and result aggregation
        """
        transformed_batches = TTA.transform_batch(tensor)

        transformed_evaluations = list()
        for transformed_batch in transformed_batches:
            if tta_crop:
                cropped = TTA.crop_batch(transformed_batch)
                cropped_evaluations = [predict(batch) for batch in cropped]
                cropped_evaluation = TTA.reverse_crop(cropped_evaluations, strategy=strategy)
                transformed_evaluations.append(cropped_evaluation)
            else:
                transformed_evaluations.append(predict(transformed_batch))

        evaluation = TTA.reverse_transform(transformed_evaluations, strategy=strategy)
        return evaluation


def _test_transform():
    batch = torch.arange(8, dtype=torch.float).reshape(2, 1, 2, 2)
    batches = TTA.transform_batch(batch)
    batch_reversed = TTA.reverse_transform(batches)
    assert len(batches) == 3 * 4
    assert not batches[0].equal(batches[1])
    assert not batches[0].equal(batches[2])
    assert not batches[0].equal(batches[3])
    assert not batches[1].equal(batches[5])
    assert not batches[1].equal(batches[9])
    assert batch.equal(batch_reversed)


def _test_crop():
    batch = torch.arange(4 * 608 * 608, dtype=torch.float).reshape(4, 1, 608, 608)
    batches = TTA.crop_batch(batch)
    batch_reversed = TTA.reverse_crop(batches)
    assert len(batches) == 4
    assert batches[0].shape[-2:] == crop_shape
    assert not batches[0].equal(batches[2])
    assert batch.equal(batch_reversed)


if __name__ == "__main__":
    # to run this test, you should ensure the following is set:
    # device = torch.device('cpu')
    _test_transform()
    _test_crop()
