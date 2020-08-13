import argparse

from rasa.cli.arguments.default_arguments import add_model_param, add_domain_param


def set_convert_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--training_data",
        required=True,
        nargs="+",
        help="Paths to the source NLU and Core data files in a Markdown format",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        nargs="+",
        help=(
            "Path to the output directory where all the converted training data files "
            "will be written to. Converted files will have the same name as the "
            "original ones with a '_converted' postfix."
        ),
    )
