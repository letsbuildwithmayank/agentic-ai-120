"""The restaurant menu and 3 tools the model can ask for.

These are plain Python functions. The model never runs them: it only asks,
and our tool loop runs them and sends the result back.

Every tool returns a dict. On a wrong input it returns {"error": "..."}
with a hint, instead of raising, so the model can fix its next request.
"""

MAX_RESULTS = 5  # search_menu never returns more than this (keeps tool results small)

MENU = [
    {
        "item_id": "paneer-tikka",
        "name": "paneer tikka",
        "price_inr": 280,
        "veg": True,
        "spicy": True,
    },
    {
        "item_id": "dal-makhani",
        "name": "dal makhani",
        "price_inr": 220,
        "veg": True,
        "spicy": False,
    },
    {
        "item_id": "butter-chicken",
        "name": "butter chicken",
        "price_inr": 340,
        "veg": False,
        "spicy": False,
    },
    {
        "item_id": "veg-biryani",
        "name": "veg biryani",
        "price_inr": 240,
        "veg": True,
        "spicy": True,
    },
    {
        "item_id": "chicken-biryani",
        "name": "chicken biryani",
        "price_inr": 300,
        "veg": False,
        "spicy": True,
    },
    {"item_id": "naan", "name": "naan", "price_inr": 40, "veg": True, "spicy": False},
    {"item_id": "coke", "name": "coke", "price_inr": 50, "veg": True, "spicy": False},
    {"item_id": "lassi", "name": "lassi", "price_inr": 60, "veg": True, "spicy": False},
]


def find_item(item_id):
    """Return the menu item with this id, or None if there is no such item."""
    for item in MENU:
        if item["item_id"] == item_id:
            return item
    return None


def search_menu(query):
    """Find items whose name contains the query, e.g. 'biryani' -> both biryanis."""
    query = query.strip().lower()
    matches = []
    for item in MENU:
        if query in item["name"]:
            matches.append(dict(item))  # a copy, so nobody can change MENU by mistake

    if not matches:
        return {
            "results": [],
            "message": f"No menu item matches '{query}'. This item is not on the menu.",
        }

    result = {"results": matches[:MAX_RESULTS]}
    if len(matches) > MAX_RESULTS:
        result["message"] = (
            f"Showing {MAX_RESULTS} of {len(matches)} items. Use a more specific query."
        )
    return result


def get_price(item_id):
    """Price of one item, by its exact item_id (like 'naan')."""
    item = find_item(item_id)
    if item is None:
        return {
            "error": f"Unknown item_id '{item_id}'. Use search_menu(query) to find the right item_id."
        }
    return {
        "item_id": item["item_id"],
        "name": item["name"],
        "price_inr": item["price_inr"],
    }


def calc_bill(items):
    """Total bill. items = [{"item_id": "naan", "qty": 2}, ...]"""
    unknown = [
        entry["item_id"] for entry in items if find_item(entry["item_id"]) is None
    ]
    if unknown:
        return {
            "error": f"Unknown item_id(s): {unknown}. Use search_menu(query) to find the right item_id."
        }

    lines = []
    total = 0
    for entry in items:
        item = find_item(entry["item_id"])
        line_total = item["price_inr"] * entry["qty"]
        lines.append(
            {
                "name": item["name"],
                "qty": entry["qty"],
                "price_inr": item["price_inr"],
                "line_total": line_total,
            }
        )
        total += line_total
    return {"lines": lines, "total_inr": total}


if __name__ == "__main__":
    # Try every tool by hand, the same way the tool loop will call them later
    print(search_menu("biryani"))
    print(search_menu("pizza"))
    print(search_menu(""))
    print(get_price("naan"))
    print(get_price("pizza-99"))
    print(
        calc_bill(
            [{"item_id": "butter-chicken", "qty": 2}, {"item_id": "naan", "qty": 4}]
        )
    )
    print(calc_bill([{"item_id": "naan", "qty": 2}, {"item_id": "pizza", "qty": 1}]))
