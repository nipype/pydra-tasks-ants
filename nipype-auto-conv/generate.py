from pathlib import Path
import sys
from click.testing import CliRunner
from nipype2pydra.cli import convert


spec_dir = Path(__file__).parent / "specs"
conv_dir = Path(__file__).parent.parent / "convert"

runner = CliRunner()
result = runner.invoke(
    convert,
    args=[str(spec_dir), str(conv_dir)],
    catch_exceptions=False,
)
