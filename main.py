import stable_pretraining as spt


if __name__=='__main__':

    train_dataset = spt.data.HFDataset(
        path="matthieulel/galaxy10_decals",
        split="train"
    )

    print(f"Dataset length: {len(train_dataset)}")
    print(f"Column names: {train_dataset.column_names}")