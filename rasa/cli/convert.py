import argparse
import asyncio
import os
from pathlib import Path
from typing import List

import rasa.cli.arguments.convert as convert_artguments
from rasa.cli.utils import (
    print_success,
    print_error_and_exit,
    print_info,
    print_warning,
)
from rasa.core.interpreter import RegexInterpreter
from rasa.core.training.story_reader.markdown_story_reader import MarkdownStoryReader
from rasa.core.training.story_writer.yaml_story_writer import YAMLStoryWriter
from rasa.nlu.training_data.formats import MarkdownReader
from rasa.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter

CONVERTED_FILE_POSTFIX = "_converted.yml"


def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    convert_parser = subparsers.add_parser(
        "convert",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=(
            "Converts NLU and Core training data files from Markdown to YAML format."
        ),
    )
    convert_parser.set_defaults(func=convert)

    convert_artguments.set_convert_arguments(convert_parser)


def convert(args: argparse.Namespace):

    output = Path(args.output[0])
    if not os.path.exists(output):
        print_error_and_exit(
            f"The output path {output} doesn't exist. Please make sure to specify "
            f"existing directory and try again."
        )
        return

    for training_data_path in args.training_data:
        if not os.path.exists(training_data_path):
            print_error_and_exit(
                f"The training data path {training_data_path} doesn't exist "
                f"and will be skipped."
            )

        loop = asyncio.get_event_loop()

        num_of_files_converted = 0
        for file in os.listdir(training_data_path):
            source_path = Path(training_data_path) / file
            output_path = Path(output) / f"{source_path.stem}{CONVERTED_FILE_POSTFIX}"

            if MarkdownReader.is_markdown_nlu_file(source_path):
                convert_nlu(source_path, output_path, source_path)
                num_of_files_converted += 1
            elif MarkdownStoryReader.is_markdown_story_file(source_path):
                loop.run_until_complete(
                    convert_core(source_path, output_path, source_path)
                )
                num_of_files_converted += 1
            else:
                print_warning(
                    f"Skipped file '{source_path}' since it's neither NLU "
                    "nor Core training data file."
                )

        print_info(f"Converted {num_of_files_converted} files, saved in '{output}'")


def convert_nlu(training_data_path: Path, output_path: Path, source_path: Path):
    reader = MarkdownReader()
    writer = RasaYAMLWriter()

    training_data = reader.read(training_data_path)
    writer.dump(output_path, training_data)

    print_success(f"Converted NLU file: '{source_path}' >> '{output_path}'")


async def convert_core(training_data_path: Path, output_path: Path, source_path: Path):
    reader = MarkdownStoryReader(RegexInterpreter())
    writer = YAMLStoryWriter()

    steps = await reader.read_from_file(training_data_path)
    writer.dump(output_path, steps)

    print_success(f"Converted Core file: '{source_path}' >> '{output_path}'")
