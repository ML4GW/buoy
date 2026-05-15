class BuoyModel:
    """
    Base class for Aframe and Amplfi models.

    Provides a shared `update_config` implementation that
    updates attributes and re-runs preprocessing setup.
    Subclasses must implement `configure_preprocessing`.
    """

    def configure_preprocessing(self) -> None:
        raise NotImplementedError

    def update_config(self, **kwargs: object) -> None:
        """
        Update configuration parameters and reconfigure preprocessing.

        Warning: some changes may not be sensible given how the model
        was trained (e.g., kernel_length, sample_rate). Changing these
        parameters may lead to unexpected results.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
        self.configure_preprocessing()
