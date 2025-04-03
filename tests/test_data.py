import torch

from data import create_dataloader_v1


def test_dataloader(sample_text):
    dataloader = create_dataloader_v1(
        sample_text,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    first_batch = next(iter(dataloader))
    input_ids, target_ids = first_batch

    assert len(input_ids) == 256
    assert len(target_ids) == 256
    assert input_ids[0].dtype == torch.int64
    assert target_ids[0].dtype == torch.int64
