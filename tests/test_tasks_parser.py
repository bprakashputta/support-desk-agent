import importlib.util

from conftest import ROOT

spec = importlib.util.spec_from_file_location("dl", ROOT / "scripts" / "download_tau_bench.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

SRC = """
from tau_bench.types import Task, Action
TASKS_TEST = [
    Task(annotator="0", user_id="u_1", instruction="do it",
         actions=[Action(name="get_order_details", kwargs={"order_id": "#W1"})], outputs=["42"]),
]
"""


def test_parse_tasks_without_executing():
    tasks = dl.parse_tasks(SRC)
    assert tasks == [
        {
            "annotator": "0",
            "user_id": "u_1",
            "instruction": "do it",
            "actions": [{"name": "get_order_details", "kwargs": {"order_id": "#W1"}}],
            "outputs": ["42"],
        }
    ]


def test_parse_rejects_unexpected_calls():
    import pytest

    with pytest.raises(ValueError):
        dl.parse_tasks("X = [Task(x=__import__('os').system('echo hi'))]")
