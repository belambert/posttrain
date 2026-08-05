"""Loading the Tulu 3 SFT mixture."""

from datasets import Dataset, load_dataset


def load_sft(
    name: str,
    n: int | None = None,
    eval_n: int = 500,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    """Load the SFT mixture as (train, eval); `n` subsamples the full ~939k rows."""
    ds = (
        load_dataset(name, split="train")
        .select_columns(["messages"])
        .shuffle(seed=seed)
    )
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    split = ds.train_test_split(test_size=min(eval_n, len(ds) // 10), seed=seed)
    return split["train"], split["test"]
