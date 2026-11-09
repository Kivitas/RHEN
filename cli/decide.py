"""rhen-decide: Evaluate decisions from command line or stdin on a given state and typed questions.

Evaluates typed non-autoregressive decision primitives concurrently in a single forward pass:
- Noul: Binary semantic truth / proposition evaluation
- Choice: Categorical 1-of-K selection
- Score: Ordinal / continuous level scoring
- MultiChoice: Independent subset selection
- Range: Bounded continuous interval regression
- Extract: Non-autoregressive span pointer extraction
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from rhen.research import device as devmod
from rhen.research.checkpointing import load_model_from_checkpoint
from rhen.research.presets import preset_config
try:
    from rhen.config import EOSConfig
    from rhen.model import EOSModel
except (ImportError, ModuleNotFoundError):
    from config import EOSConfig
    from model import EOSModel

from rhen import (
    Choice,
    ChoiceResult,
    Extract,
    ExtractResult,
    MultiChoice,
    MultiChoiceResult,
    Noul,
    NoulResult,
    Range,
    RangeResult,
    RhenEngine,
    Score,
    ScoreResult,
)


def _parse_state_arg(state_arg: Optional[str]) -> Any:
    """Parse state from CLI argument (JSON string, @file path, or text)."""
    if not state_arg:
        return None
    if state_arg.startswith("@"):
        path = state_arg[1:]
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    if os.path.isfile(state_arg):
        with open(state_arg, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    try:
        return json.loads(state_arg)
    except (json.JSONDecodeError, ValueError):
        return state_arg


def _dict_to_primitive(item: Dict[str, Any]) -> Any:
    """Convert question dict to corresponding RHEN typed primitive."""
    kind = item.get("type", item.get("primitive", "choice")).lower()
    question = item.get("question", item.get("proposition", ""))
    if kind == "noul":
        return Noul(proposition=question or item.get("text", ""))
    elif kind == "choice":
        options = item.get("options", [])
        return Choice(question=question, options=options)
    elif kind == "score":
        levels = item.get("levels", [])
        min_s = float(item.get("min_score", 0.0))
        max_s = float(item.get("max_score", 10.0))
        return Score(question=question, levels=levels, min_score=min_s, max_score=max_s)
    elif kind == "multichoice":
        options = item.get("options", [])
        thresh = float(item.get("threshold", 0.5))
        return MultiChoice(question=question, options=options, threshold=thresh)
    elif kind == "range":
        min_v = float(item.get("min_val", 0.0))
        max_v = float(item.get("max_val", 100.0))
        return Range(question=question, min_val=min_v, max_val=max_v)
    elif kind == "extract":
        slot = item.get("slot_name", item.get("slot", question))
        query = item.get("query", "")
        return Extract(slot_name=slot, query=query)
    else:
        raise ValueError(f"Unknown primitive type: {kind}")


def _serialize_result(res: Any, engine: Optional[RhenEngine] = None, check_abstention: bool = True) -> Dict[str, Any]:
    """Serialize a typed primitive result into a clean dictionary."""
    data: Dict[str, Any] = {}
    free_energy = getattr(res, "free_energy", None)
    conformal_set = getattr(res, "conformal_set", None)

    if isinstance(res, NoulResult):
        data = {
            "primitive": "noul",
            "proposition": res.proposition,
            "decision": bool(res.decision),
            "p_true": float(res.p_true),
            "confidence": float(res.confidence),
        }
    elif isinstance(res, ChoiceResult):
        data = {
            "primitive": "choice",
            "question": res.question,
            "choice_label": str(res.choice_label),
            "choice_index": int(res.choice_index),
            "confidence": float(res.confidence),
            "probabilities": res.probabilities,
        }
    elif isinstance(res, ScoreResult):
        data = {
            "primitive": "score",
            "question": res.question,
            "score": float(res.score),
            "confidence": float(res.confidence),
            "level_probabilities": res.level_probabilities,
        }
    elif isinstance(res, MultiChoiceResult):
        data = {
            "primitive": "multichoice",
            "question": res.question,
            "selected_options": list(res.selected_options),
            "confidence": float(res.confidence),
            "probabilities": res.probabilities,
        }
    elif isinstance(res, RangeResult):
        data = {
            "primitive": "range",
            "question": res.question,
            "expected_value": float(res.expected_value),
            "interval_low": float(res.interval_low),
            "interval_high": float(res.interval_high),
            "confidence": float(res.confidence),
        }
    elif isinstance(res, ExtractResult):
        data = {
            "primitive": "extract",
            "slot_name": res.slot_name,
            "extracted_value": str(res.extracted_value),
            "start_idx": int(res.start_idx),
            "end_idx": int(res.end_idx),
            "confidence": float(res.confidence),
        }
    elif hasattr(res, "__dict__"):
        data = dict(res.__dict__)
    else:
        data = {"result": str(res)}

    if free_energy is not None:
        data["free_energy"] = float(free_energy)
    if conformal_set is not None:
        data["conformal_set"] = conformal_set

    if check_abstention and engine is not None:
        probs = getattr(res, "probabilities", None)
        if probs is None and hasattr(res, "p_true"):
            probs = [float(res.p_true), 1.0 - float(res.p_true)]
        verdict = engine.abstention_verdict(
            probabilities=probs,
            free_energy=free_energy,
            conformal_set=conformal_set,
        )
        data["abstention"] = {
            "action": verdict.decision,
            "reasons": verdict.reasons,
        }

    return data


def format_table(results: List[Dict[str, Any]]) -> str:
    """Format evaluation results into a clean ASCII table."""
    lines = []
    header = f"{'PRIMITIVE':<12} | {'QUESTION / PROPOSITION':<32} | {'DECISION / VALUE':<22} | {'CONF':<6} | {'ACTION':<7}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        prim = r.get("primitive", "-")
        q = r.get("question") or r.get("proposition") or r.get("slot_name") or "-"
        if len(q) > 30:
            q = q[:27] + "..."
        dec = (
            str(r.get("choice_label"))
            if "choice_label" in r
            else str(r.get("decision"))
            if "decision" in r
            else f"{r.get('score'):.2f}"
            if "score" in r
            else str(r.get("selected_options"))
            if "selected_options" in r
            else f"{r.get('expected_value'):.2f}"
            if "expected_value" in r
            else str(r.get("extracted_value", "-"))
        )
        if len(dec) > 20:
            dec = dec[:17] + "..."
        conf = f"{r.get('confidence', 0.0):.2f}"
        act = r.get("abstention", {}).get("action", "ANSWER")
        lines.append(f"{prim:<12} | {q:<32} | {dec:<22} | {conf:<6} | {act:<7}")
    return "\n".join(lines)


class _QuestionAction(argparse.Action):
    """Accumulate typed questions in the exact order they appear on the command line."""

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, "ordered_questions", None)
        if items is None:
            items = []
            setattr(namespace, "ordered_questions", items)

        if option_string == "--noul":
            items.append(Noul(proposition=values))
        elif option_string == "--choice":
            if len(values) < 2:
                parser.error("Choice requires at least question and 2 options: --choice 'Q?' opt1 opt2 ...")
            items.append(Choice(question=values[0], options=values[1:]))
        elif option_string == "--score":
            levels = values[1:] if len(values) > 1 else []
            items.append(Score(question=values[0], levels=levels))
        elif option_string == "--multichoice":
            items.append(MultiChoice(question=values[0], options=values[1:]))
        elif option_string == "--range":
            min_v = float(values[1]) if len(values) > 1 else 0.0
            max_v = float(values[2]) if len(values) > 2 else 100.0
            items.append(Range(question=values[0], min_val=min_v, max_val=max_v))
        elif option_string == "--extract":
            slot = values[0]
            query = values[1] if len(values) > 1 else ""
            items.append(Extract(slot_name=slot, query=query))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rhen-decide",
        description="Evaluate non-autoregressive decisions on structured state and typed questions.",
    )
    # State specification
    parser.add_argument(
        "-s",
        "--state",
        default=None,
        help="Structured state: JSON string, @file path, path to state file, or '-' for stdin.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Force reading state or JSON request from standard input.",
    )

    # Typed questions CLI flags
    parser.add_argument(
        "--noul",
        action=_QuestionAction,
        metavar="PROPOSITION",
        help="Noul: Evaluate binary truth of proposition (can be repeated).",
    )
    parser.add_argument(
        "--choice",
        action=_QuestionAction,
        nargs="+",
        metavar=("QUESTION", "OPTION"),
        help="Choice: 1-of-K selection. Usage: --choice 'Question?' opt1 opt2 ... (can be repeated).",
    )
    parser.add_argument(
        "--score",
        action=_QuestionAction,
        nargs="+",
        metavar="ITEM",
        help="Score: Continuous/ordinal ranking. Usage: --score 'Risk level' [level1 level2 ...]",
    )
    parser.add_argument(
        "--multichoice",
        action=_QuestionAction,
        nargs="+",
        metavar=("QUESTION", "OPTION"),
        help="MultiChoice: Subset selection. Usage: --multichoice 'Checks?' check1 check2 ...",
    )
    parser.add_argument(
        "--range",
        action=_QuestionAction,
        nargs="+",
        metavar="ITEM",
        help="Range: Interval regression. Usage: --range 'Latency (ms)' [min] [max]",
    )
    parser.add_argument(
        "--extract",
        action=_QuestionAction,
        nargs="+",
        metavar=("SLOT_NAME", "QUERY"),
        help="Extract: Span extraction. Usage: --extract slot_name [query]",
    )
    parser.add_argument(
        "-q",
        "--questions-file",
        default=None,
        help="Path to JSON file containing question definitions array.",
    )
    parser.add_argument(
        "--questions-json",
        default=None,
        help="Inline JSON array of question definitions.",
    )

    # Model and inference options
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Path to trained checkpoint directory containing model.pt and model_config.json.",
    )
    parser.add_argument(
        "--preset",
        default="tiny",
        choices=["tiny", "small", "base"],
        help="Model preset if no checkpoint is given (default: tiny).",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        help="Target device: auto, cpu, cuda, cuda:0, etc. (default: auto).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name or directory path (default: bundled EOS tokenizer).",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Recurrent loop count (default: config default).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Conformal prediction alpha error rate (default: 0.05).",
    )
    parser.add_argument(
        "--ood-threshold",
        type=float,
        default=4.5,
        help="Free energy threshold for out-of-distribution detection (default: 4.5).",
    )
    parser.add_argument(
        "--no-abstain-check",
        action="store_true",
        help="Disable geometric abstention checking.",
    )

    # Output options
    parser.add_argument(
        "-f",
        "--format",
        default="json",
        choices=["json", "pretty", "table", "text"],
        help="Output format: json, pretty, table, or text (default: json).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="File path to write evaluation results (default: stdout).",
    )
    return parser


def build_questions_from_args(args: argparse.Namespace) -> List[Any]:
    """Parse typed questions from command-line arguments in order."""
    questions: List[Any] = list(getattr(args, "ordered_questions", []) or [])

    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as fh:
            items = json.load(fh)
            for item in items:
                questions.append(_dict_to_primitive(item))

    if args.questions_json:
        items = json.loads(args.questions_json)
        for item in items:
            questions.append(_dict_to_primitive(item))

    return questions


def initialize_engine(args: argparse.Namespace) -> RhenEngine:
    """Initialize RhenEngine using either a checkpoint or a lightweight preset."""
    device = devmod.resolve_device(args.device)
    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint, device=device)
    else:
        cfg = EOSConfig(**preset_config(args.preset))
        model = EOSModel(cfg).to(device)

    tokenizer = None
    if args.tokenizer:
        from tokenizer import get_tokenizer

        tokenizer = get_tokenizer(args.tokenizer)

    engine = RhenEngine(model=model, tokenizer=tokenizer, device=device)
    engine.conformal_engine.alpha = args.alpha
    engine.ood_detector.ood_threshold = args.ood_threshold
    return engine


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    state: Any = None
    stdin_data = None

    if args.stdin or args.state == "-" or (not args.state and not sys.stdin.isatty()):
        try:
            stdin_data = sys.stdin.read().strip()
        except Exception:
            stdin_data = None

    # Determine state and questions
    questions = build_questions_from_args(args)

    if stdin_data:
        # Check if stdin is a JSON object with state and questions
        try:
            parsed = json.loads(stdin_data)
            if isinstance(parsed, dict):
                if "state" in parsed:
                    state = parsed["state"]
                if "questions" in parsed and not questions:
                    for q in parsed["questions"]:
                        questions.append(_dict_to_primitive(q))
            elif isinstance(parsed, list) and not state:
                state = parsed
        except json.JSONDecodeError:
            if not state:
                state = stdin_data

    if state is None and args.state and args.state != "-":
        state = _parse_state_arg(args.state)

    if state is None:
        parser.error("No state provided. Provide state via --state, @filename, or stdin.")

    if not questions:
        parser.error("No questions specified. Use --choice, --noul, --score, --multichoice, or stdin JSON.")

    engine = initialize_engine(args)

    # Perform single forward-pass evaluation
    start_t = time.perf_counter()
    raw_results = engine.evaluate(state, questions, num_loops=args.num_loops)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    serialized = [
        _serialize_result(r, engine=engine, check_abstention=not args.no_abstain_check)
        for r in raw_results
    ]

    output_payload: Dict[str, Any] = {
        "results": serialized,
        "elapsed_ms": round(elapsed_ms, 2),
        "device": str(engine.device),
        "num_questions": len(serialized),
    }

    # Format output
    out_text = ""
    if args.format == "json":
        out_text = json.dumps(output_payload)
    elif args.format == "pretty":
        out_text = json.dumps(output_payload, indent=2)
    elif args.format == "table":
        table_str = format_table(serialized)
        out_text = f"{table_str}\n\nEvaluated {len(serialized)} decision(s) in {elapsed_ms:.1f}ms on {engine.device}"
    elif args.format == "text":
        lines = []
        for r in serialized:
            lines.append(f"[{r['primitive'].upper()}] {r.get('question') or r.get('proposition')}")
            dec = r.get("choice_label") or r.get("decision") or r.get("score") or r.get("selected_options") or r.get("expected_value")
            lines.append(f"  Decision: {dec} (confidence: {r.get('confidence', 0.0):.4f})")
            if "abstention" in r:
                lines.append(f"  Action:   {r['abstention']['action']} (reasons: {r['abstention']['reasons']})")
        lines.append(f"Latency: {elapsed_ms:.1f}ms")
        out_text = "\n".join(lines)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out_text + "\n")
    else:
        print(out_text)

    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
