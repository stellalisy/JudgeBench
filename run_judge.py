from typing import List, Dict, Any
import argparse
import asyncio
import json
import os
import random

from tqdm.asyncio import tqdm_asyncio

import utils.file_operations as file_operations
import utils.judges as judges
import utils.metrics as metrics


async def judge_pairs(pairs: List[Dict[str, Any]], judge, concurrency_limit: int = 1, reverse_order: int = False, output_file: str = None):
    semaphore = asyncio.Semaphore(concurrency_limit)
    file_lock = asyncio.Lock()
    is_rubric_judge = isinstance(judge, judges.RubricJudge)
    
    async def judge_pair(pair: Dict[str, Any]):
        async with semaphore:
            
            question = pair["question"]
            response_A = pair["response_A"]
            response_B = pair["response_B"]

            shared_rubric = None
            if is_rubric_judge and reverse_order:
                shared_rubric = await judge.generate_rubric(question)

            try:
                judgment_1 = await judge.get_judgment(question, response_A, response_B,
                                                      rubric=shared_rubric)
            except Exception as e:
                print(f"Failed to judge pair {pair['pair_id']} due to the following error: {e}.")
                judgment_1 = None
            judgments = [judgment_1]
            
            if reverse_order:
                try:
                    judgment_2 = await judge.get_judgment(question, response_B, response_A,
                                                          rubric=shared_rubric)
                except Exception as e:
                    print(f"Failed to judge pair {pair['pair_id']} due to the following error: {e}.")
                    judgment_2 = None
                judgments.append(judgment_2)
            
            pair["judge_name"] = getattr(judge, 'judge_model_name', str(type(judge).__name__))
            pair["judgments"] = judgments
            return pair

    tasks = [asyncio.create_task(judge_pair(pair)) for pair in pairs]

    for future in tqdm_asyncio.as_completed(tasks):
        pair = await future
        if output_file is not None:
            async with file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    return pairs


def main(args: argparse.Namespace) -> None:
    
    random.seed(args.seed)
    
    pairs = file_operations.read_jsonl(args.pairs)    

    dataset_name = os.path.basename(args.pairs).replace(".jsonl", "")

    if args.rubric_model:
        rubric_model_tag = os.path.basename(args.rubric_model.rstrip("/")).replace("/", "-")
    else:
        rubric_model_tag = "none"
    judge_model_tag = args.judge_model.replace("/", "-")
    file_name = (
        f"dataset=judgebench,"
        f"response_model=gpt-4o-2024-05-13,"
        f"judge_name={args.judge_name},"
        f"rubric_model={rubric_model_tag},"
        f"judge_model={judge_model_tag}.jsonl"
    )

    output_dir = args.output_dir or "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping judging pairs...")
        original_num_pairs = len(pairs)
        existing_pairs = file_operations.read_jsonl(file_path)
        existing_pair_ids = {pair["pair_id"] for pair in existing_pairs}
        pairs = [pair for pair in pairs if pair["pair_id"] not in existing_pair_ids]
        print(f"Skipped {original_num_pairs - len(pairs)} pairs.")

    judge_kwargs = {}
    if args.judge_name == "rubric":
        judge_kwargs["rubric_model_name"] = args.rubric_model
        judge_kwargs["rubric_port"] = args.rubric_port
        judge_kwargs["judge_port"] = args.judge_port
    elif args.judge_name == "no_rubric":
        judge_kwargs["judge_port"] = args.judge_port

    judge = judges.get_judge_from_judge_name_and_model(
        args.judge_name, args.judge_model, **judge_kwargs
    )

    if pairs: 
        print("Judging pairs ...")
        pairs = asyncio.run(
            judge_pairs(
                pairs,
                judge,
                reverse_order=not args.single_game,
                concurrency_limit=args.concurrency_limit,
                output_file=file_path,
            )
        )

    print("Computing final metrics ...") 
    pairs = file_operations.read_jsonl(file_path)
    for source in ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]:
        score = metrics.compute_final_metrics(pairs, not args.single_game, include_fn = lambda x: x["source"].startswith(source))
        print(f"{source if source else 'Overall'}: {score:.2f}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge_name', type=str, required=True)
    parser.add_argument('--judge_model', type=str, required=True)
    parser.add_argument('--rubric_model', type=str, default=None)
    parser.add_argument('--response_model', type=str, default=None)
    parser.add_argument('--rubric_port', type=int, default=8000)
    parser.add_argument('--judge_port', type=int, default=8001)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--single_game', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--concurrency_limit', type=int, default=1)
    parser.add_argument('--pairs', type=str, required=True)
    args = parser.parse_args()
    main(args)
