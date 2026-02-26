from datasets import load_dataset

def get_dataset(dataset_name="DanielHesslow/SwissProt-EC"):
    """
    :param dataset_name:String
    :return: dataset with train, test and val datasets
    """

    # Example: Loading a SwissProt subset with Pfam labels
    dataset = load_dataset(dataset_name)
    # You can access the data splits if available (often just 'train')
    train_data = dataset["train"]
    test_data = dataset["test"]


    return train_data, test_data