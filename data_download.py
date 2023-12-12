from datasets import load_dataset, load_from_disk

d = load_dataset("super_glue", "sst2")
d.save_to_disk("data/sst2")
