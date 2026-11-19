"""What the model sees about each tool (TOOLS), and what our code runs (TOOL_FUNCTIONS).

The model only reads the name, the description and the parameters (JSON Schema).
It never sees the Python code. So the description is the only manual it gets:
it must say what the tool does, WHEN to use it, and the exact format of each parameter.
"""

import json

from agentic_ai_120.menu_tools import calc_bill, get_price, search_menu

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": (
                "Search the restaurant menu by dish name. Use it to find an item_id, "
                "or to check if a dish is on the menu, veg, or spicy. "
                "Returns at most 5 items, each with item_id, name, price_inr, veg and spicy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Part of a dish name in English, e.g. 'biryani' or 'naan'. Fix spelling first ('panner' -> 'paneer').",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get the price in INR of ONE menu item. Needs the exact item_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Exact item_id from search_menu, e.g. 'naan' or 'butter-chicken'. Never guess it.",
                    },
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_bill",
            "description": (
                "Calculate the total bill for an order of one or more items. "
                "It looks up the prices itself, so there is no need to call get_price first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "Every item in the order with its quantity.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_id": {
                                    "type": "string",
                                    "description": "Exact item_id from search_menu.",
                                },
                                "qty": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "description": "How many, at least 1.",
                                },
                            },
                            "required": ["item_id", "qty"],
                        },
                    },
                },
                "required": ["items"],
            },
        },
    },
]

# Tool name (what the model asks for) -> Python function (what our code runs)
TOOL_FUNCTIONS = {
    "search_menu": search_menu,
    "get_price": get_price,
    "calc_bill": calc_bill,
}


if __name__ == "__main__":
    # Every tool the model can see must have a function we can run, and the other way round
    schema_names = [tool["function"]["name"] for tool in TOOLS]
    assert sorted(schema_names) == sorted(TOOL_FUNCTIONS), (
        "TOOLS and TOOL_FUNCTIONS don't match"
    )
    print("Tools the model will see:", schema_names)
    print("\nOne schema, exactly as it is sent to the model:")
    print(json.dumps(TOOLS[2], indent=2))
