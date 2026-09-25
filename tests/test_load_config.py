"""Test loading of configuration files."""

from pathlib import Path

from plasmid_net.train import load_config, ConfigTrain


def test_load_config():
    """Test the loading of the configuration file."""
    config = load_config(Path("training_config.yaml"))
    assert config is not None
    assert isinstance(config, ConfigTrain)
