from typer.testing import CliRunner

from posttrain.cli import app

runner = CliRunner()


def test_help_needs_no_training_deps():
    assert runner.invoke(app, ["sft", "--help"]).exit_code == 0


def test_bad_override_is_reported():
    assert runner.invoke(app, ["sft", "-s", "nope=1"]).exit_code != 0
