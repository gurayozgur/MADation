from .loaders import FaceDataset, TestFaceDataset

def get_dataset(rank, **kwargs):
    dataset_name = kwargs["dataset_name"]

    if dataset_name == 'MADTrain':
        trainset = FaceDataset(kwargs["dataset_path"], is_train=True)
        # config.test_dataset_path = [dataset_paths["facemorpher"], dataset_paths["mipgan1"], dataset_paths["mipgan2"], dataset_paths["mordiff"], dataset_paths["opencv"], dataset_paths["webmorph"]]
        # create a list of test datasets
        testset = [TestFaceDataset(path, is_train=False) for path in kwargs["test_dataset_path"]]
        return trainset, testset
    else:
        raise ValueError()

