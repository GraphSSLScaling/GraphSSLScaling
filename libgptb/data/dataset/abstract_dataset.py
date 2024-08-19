class AbstractDataset(object):

    def __init__(self, config):
        raise NotImplementedError("Dataset not implemented")

    def get_data(self):
        """

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        raise NotImplementedError("get_data not implemented")

    def get_data_feature(self):
        """
        Returns:
            dict: dataset's feature
        """
        raise NotImplementedError("get_data_feature not implemented")