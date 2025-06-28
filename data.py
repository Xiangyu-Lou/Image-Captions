from datasets import load_dataset

def add_prefix_to_file_name(batch):
    batch["file_name"] = ["data/" + file_name for file_name in batch["file_name"]]
    return batch

def loading_dataset(dataset_name='phiyodr/coco2017', sample_index=None):
    print('----Loading Data----')
    
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    dataset = dataset.map(    # type: ignore
        add_prefix_to_file_name,
        batched=True,
        batch_size=1000
    )
    
    if sample_index != None:
        sample = dataset['train'][sample_index] # type: ignore
        print(f"File Name: {sample['file_name']}")
        print(f"URL: {sample['coco_url']}")
        print(f"Size: {sample['height']} x {sample['width']}")
        
    return dataset
    
if __name__ == '__main__':
    loading_dataset(sample_index=1)