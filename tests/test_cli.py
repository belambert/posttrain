import pytest
from typer.testing import CliRunner

from posttrain.cli import app

runner = CliRunner()


@pytest.mark.parametrize("cmd", ["sft", "check", "chat"])
def test_help_needs_no_training_deps(cmd):
    assert runner.invoke(app, [cmd, "--help"]).exit_code == 0


def test_sft_is_a_named_subcommand():
    """--help exits 0 even on a bad command line, so route a real argument."""
    result = runner.invoke(app, ["sft", "-s", "nope=1"])
    assert "unexpected extra argument" not in result.output.lower()
    assert isinstance(result.exception, ValueError)
    assert "nope" in str(result.exception)
