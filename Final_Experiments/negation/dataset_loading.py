import datasets as ds

dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2014,
    coco_task="instances",
)
trainset = dataset["train"]
