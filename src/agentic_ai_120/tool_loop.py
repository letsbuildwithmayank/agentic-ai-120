"""The tool-call loop, written by hand (no framework).

ask the model (with TOOLS)
  -> it answers with text?      done
  -> it asks for tools?         check the arguments (Pydantic), run each tool,
                                add each result with its tool_call_id, ask again
stop after MAX_ROUNDS rounds, so a confused model can't loop forever
"""

import json
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from agentic_ai_120.llm_client import chat
from agentic_ai_120.menu_tool_schemas import TOOL_FUNCTIONS, TOOLS

MODEL = "groq/openai/gpt-oss-20b"
MAX_ROUNDS = 6  # most questions need 1-3 rounds; more means the model is stuck
MAX_RESULT_CHARS = (
    2000  # a tool result longer than this is cut before it goes to the model
)

SYSTEM_PROMPT = """You are the order assistant of the Flexy World restaurant.
Answer questions about the menu, prices and bills by using the tools.
Never guess a price or an item_id: always get them from the tools.
If a dish is not on the menu, say so politely and suggest something that is.
Reply in the customer's language (English or Hinglish), short and friendly."""


# The exact shape of each tool's arguments. The model's JSON is checked against these
# BEFORE the tool runs, so a wrong argument becomes a clear error, not a crash.
class SearchMenuArgs(BaseModel):
    query: str


class GetPriceArgs(BaseModel):
    item_id: str


class BillItem(BaseModel):
    item_id: str
    qty: int = Field(ge=1)  # at least 1


class CalcBillArgs(BaseModel):
    items: list[BillItem]


ARG_MODELS = {
    "search_menu": SearchMenuArgs,
    "get_price": GetPriceArgs,
    "calc_bill": CalcBillArgs,
}


def run_tool(name, arguments_text):
    """Run one tool request. A model mistake never raises: it comes back as {"error": ...}."""
    if name not in TOOL_FUNCTIONS:
        return {
            "error": f"Unknown tool '{name}'. Available tools: {list(TOOL_FUNCTIONS)}"
        }

    try:
        raw_args = json.loads(arguments_text or "{}")
    except json.JSONDecodeError:
        return {"error": f"The arguments are not valid JSON: {arguments_text!r}"}

    try:
        args = ARG_MODELS[name].model_validate(raw_args)
    except ValidationError as error:
        problems = []
        for problem in error.errors():
            where = ".".join(str(part) for part in problem["loc"])  # e.g. "items.0.qty"
            problems.append(f"{where}: {problem['msg']}")
        return {"error": f"Wrong arguments for {name}: " + "; ".join(problems)}

    return TOOL_FUNCTIONS[name](**args.model_dump())


def to_tool_content(result):
    """The tool result as text for the model, cut if it is too long."""
    text = json.dumps(result, ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " ...[cut: result too long]"
    return text


def assistant_to_dict(message):
    """Turn LiteLLM's message object into a plain dict we can send back in the next round."""
    result = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = []
        for call in message.tool_calls:
            result["tool_calls"].append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
    return result


def short(text, limit=110):
    """Shorten long text for the printed trace."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def run_agent(question, model=MODEL, tools=TOOLS, max_rounds=MAX_ROUNDS):
    """Let the model use tools until it answers. Returns (answer, trace)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []
    print(f"\nQ: {question}")

    for round_number in range(1, max_rounds + 1):
        message, model_used = chat(
            messages, model=model, tools=tools, temperature=0, max_retries=4
        )
        messages.append(
            assistant_to_dict(message)
        )  # the model's requests must stay in the history

        if not message.tool_calls:  # no requests: this is the final answer
            print(f"  Round {round_number}: answer")
            print(f"A: {message.content}")
            trace.append(
                {"round": round_number, "type": "answer", "text": message.content}
            )
            return message.content, trace

        print(
            f"  Round {round_number}: {len(message.tool_calls)} tool call(s)  [{model_used}]"
        )
        for call in message.tool_calls:  # one request or several (parallel calls)
            result = run_tool(call.function.name, call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,  # copied from the request, so the model can match them
                    "content": to_tool_content(result),
                }
            )
            print(f"    {call.function.name}({call.function.arguments})")
            print(f"      -> {short(json.dumps(result, ensure_ascii=False))}")
            trace.append(
                {
                    "round": round_number,
                    "type": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                    "result": result,
                }
            )

    # Out of rounds: ask once more with tools switched off, so the model must answer with what it has
    print(
        f"  Stopped after {max_rounds} rounds: asking for a final answer without tools"
    )
    message, model_used = chat(
        messages, model=model, tools=tools, tool_choice="none", temperature=0
    )
    answer = message.content or "Sorry, I could not finish this request."
    print(f"A: {answer}")
    trace.append({"round": max_rounds + 1, "type": "forced_answer", "text": answer})
    return answer, trace


if __name__ == "__main__":
    # uv run python src/agentic_ai_120/tool_loop.py "2 naan aur 1 lassi ka bill?"
    load_dotenv()  # API keys from .env
    if len(sys.argv) > 1:
        run_agent(sys.argv[1])
    else:
        run_agent("2 butter chicken aur 4 naan ka bill kitna hoga?")
