class AbstractExecutor(object):

    def __init__(self, config, model, data_feature):
        raise NotImplementedError("Executor not implemented")

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor train not implemented")

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor evaluate not implemented")

    def load_model(self, cache_name):
        """
        load model from cache_name

        Args:
            cache_name(str): name to load from
        """
        raise NotImplementedError("Executor load cache not implemented")

    def save_model(self, cache_name):
        """
        save model to cache_name

        Args:
            cache_name(str): name to save as
        """
        raise NotImplementedError("Executor save cache not implemented")