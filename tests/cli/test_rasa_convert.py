import os
from typing import Callable

from _pytest.pytester import RunResult


def test_rasa_convert(
    run_in_simple_project: Callable[..., RunResult], run: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    simple_nlu_md = """
    ## intent:greet
    - hey
    - hello
    """

    with open("data/nlu.md", "w") as f:
        f.write(simple_nlu_md)

    simple_story_md = """
    ## happy path
    * greet
        - utter_greet
    """

    with open("data/stories.md", "w") as f:
        f.write(simple_story_md)

    run_in_simple_project(
        "convert", "--training_data", "data", "--output", converted_data_folder,
    )

    assert len(os.listdir(converted_data_folder)) == 2
