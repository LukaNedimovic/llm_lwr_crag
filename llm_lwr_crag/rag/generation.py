from box import Box
from handlers.auto import AutoLLM


def setup_generation(args: Box) -> AutoLLM:
    gen_llm = None

    # Answer generation using LLM
    if args.generator:
        gen_llm = AutoLLM.from_args(args.generator)

    return gen_llm
