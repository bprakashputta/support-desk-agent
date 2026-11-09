"""Typed retail tools.

The business logic is ported from the tau-bench retail domain (MIT, Copyright Sierra;
github.com/sierra-research/tau-bench, tau_bench/envs/retail/tools). Return strings and
database mutations are kept byte-for-byte compatible so the benchmark's ground-truth
database hash still applies; `tests/test_tools_parity.py` replays every test-split
ground-truth action to check that.

What is added on top of the upstream tools:
- a pydantic model per tool: arguments are validated before any code runs, and the JSON
  schema the model sees is generated from the same class (one source of truth);
- a `kind` per tool (read / write / handoff / meta) that the policy layer, the audit log
  and the idempotency cache key off;
- `refund_amount()`: how much money a write would send back to the customer, used for the
  human-approval threshold.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .store import DataView

ORDER_ID = "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
PAYMENT_ID = (
    "The payment method id to pay or receive refund for the item price difference, such as "
    "'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."
)


class Kind(StrEnum):
    READ = "read"
    WRITE = "write"
    HANDOFF = "handoff"
    META = "meta"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FindUserIdByEmail(_Args):
    """Find user id by email. If the user is not found, the function will return an error message."""

    email: str = Field(description="The email of the user, such as 'something@example.com'.")


class FindUserIdByNameZip(_Args):
    """Find user id by first name, last name, and zip code. If the user is not found, the function will return an error message. By default, find user id by email, and only call this function if the user is not found by email or cannot remember email."""  # noqa: E501

    first_name: str = Field(description="The first name of the customer, such as 'John'.")
    last_name: str = Field(description="The last name of the customer, such as 'Doe'.")
    zip: str = Field(description="The zip code of the customer, such as '12345'.")


class GetUserDetails(_Args):
    """Get the details of a user, including their orders."""

    user_id: str = Field(description="The user id, such as 'sara_doe_496'.")


class GetOrderDetails(_Args):
    """Get the status and details of an order."""

    order_id: str = Field(description=ORDER_ID)


class GetProductDetails(_Args):
    """Get the inventory details of a product."""

    product_id: str = Field(
        description="The product id, such as '6086499569'. Be careful the product id is different from the item id."
    )


class ListAllProductTypes(_Args):
    """List the name and product id of all product types. Each product type has a variety of different items with unique item ids and options. There are only 50 product types in the store."""  # noqa: E501


class Calculate(_Args):
    """Calculate the result of a mathematical expression."""

    expression: str = Field(
        description="The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces."  # noqa: E501
    )


class Think(_Args):
    """Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."""  # noqa: E501

    thought: str = Field(description="A thought to think about.")


class CancelPendingOrder(_Args):
    """Cancel a pending order. If the order is already processed or delivered, it cannot be cancelled. The agent needs to explain the cancellation detail and ask for explicit user confirmation (yes/no) to proceed. If the user confirms, the order status will be changed to 'cancelled' and the payment will be refunded. The refund will be added to the user's gift card balance immediately if the payment was made using a gift card, otherwise the refund would take 5-7 business days to process. The function returns the order details after the cancellation."""  # noqa: E501

    order_id: str = Field(description=ORDER_ID)
    reason: Literal["no longer needed", "ordered by mistake"] = Field(
        description="The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'."
    )


class ModifyPendingOrderAddress(_Args):
    """Modify the shipping address of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed."""  # noqa: E501

    order_id: str = Field(description=ORDER_ID)
    address1: str = Field(description="The first line of the address, such as '123 Main St'.")
    address2: str = Field(description="The second line of the address, such as 'Apt 1' or ''.")
    city: str = Field(description="The city, such as 'San Francisco'.")
    state: str = Field(description="The state, such as 'CA'.")
    country: str = Field(description="The country, such as 'USA'.")
    zip: str = Field(description="The zip code, such as '12345'.")


class ModifyPendingOrderPayment(_Args):
    """Modify the payment method of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed."""  # noqa: E501

    order_id: str = Field(description=ORDER_ID)
    payment_method_id: str = Field(
        description="The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."  # noqa: E501
    )


class ModifyPendingOrderItems(_Args):
    """Modify items in a pending order to new items of the same product type. For a pending order, this function can only be called once. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed."""  # noqa: E501

    order_id: str = Field(description=ORDER_ID)
    item_ids: list[str] = Field(
        description="The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list."
    )
    new_item_ids: list[str] = Field(
        description="The item ids to be modified for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product."  # noqa: E501
    )
    payment_method_id: str = Field(description=PAYMENT_ID)


class ModifyUserAddress(_Args):
    """Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed."""  # noqa: E501

    user_id: str = Field(description="The user id, such as 'sara_doe_496'.")
    address1: str = Field(description="The first line of the address, such as '123 Main St'.")
    address2: str = Field(description="The second line of the address, such as 'Apt 1' or ''.")
    city: str = Field(description="The city, such as 'San Francisco'.")
    state: str = Field(description="The state, such as 'CA'.")
    country: str = Field(description="The country, such as 'USA'.")
    zip: str = Field(description="The zip code, such as '12345'.")


class ReturnDeliveredOrderItems(_Args):
    """Return some items of a delivered order. The order status will be changed to 'return requested'. The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed. The user will receive follow-up email for how and where to return the item."""  # noqa: E501

    order_id: str = Field(description=ORDER_ID)
    item_ids: list[str] = Field(
        description="The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list."
    )
    payment_method_id: str = Field(description=PAYMENT_ID)


class ExchangeDeliveredOrderItems(_Args):
    """Exchange items in a delivered order to new items of the same product type. For a delivered order, return or exchange can be only done once by the agent. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed."""  # noqa: E501

    order_id: str = Field(description=ORDER_ID)
    item_ids: list[str] = Field(
        description="The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list."
    )
    new_item_ids: list[str] = Field(
        description="The item ids to be exchanged for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product."  # noqa: E501
    )
    payment_method_id: str = Field(description=PAYMENT_ID)


class TransferToHumanAgents(_Args):
    """Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools."""  # noqa: E501

    summary: str = Field(description="A summary of the user's issue.")


# --------------------------------------------------------------------------- logic


def find_user_id_by_email(d: DataView, a: FindUserIdByEmail) -> str:
    return d.find_user_by_email(a.email) or "Error: user not found"


def find_user_id_by_name_zip(d: DataView, a: FindUserIdByNameZip) -> str:
    return d.find_user_by_name_zip(a.first_name, a.last_name, a.zip) or "Error: user not found"


def get_user_details(d: DataView, a: GetUserDetails) -> str:
    return json.dumps(d.users[a.user_id]) if a.user_id in d.users else "Error: user not found"


def get_order_details(d: DataView, a: GetOrderDetails) -> str:
    return json.dumps(d.orders[a.order_id]) if a.order_id in d.orders else "Error: order not found"


def get_product_details(d: DataView, a: GetProductDetails) -> str:
    if a.product_id in d.products:
        return json.dumps(d.products[a.product_id])
    return "Error: product not found"


def list_all_product_types(d: DataView, a: ListAllProductTypes) -> str:
    return json.dumps(dict(sorted(d.product_index().items())))


def calculate(d: DataView, a: Calculate) -> str:
    if not all(c in "0123456789+-*/(). " for c in a.expression):
        return "Error: invalid characters in expression"
    try:
        return str(round(float(eval(a.expression, {"__builtins__": None}, {})), 2))  # noqa: S307
    except Exception as e:  # noqa: BLE001 - mirrors upstream behaviour
        return f"Error: {e}"


def think(d: DataView, a: Think) -> str:
    return ""


def transfer_to_human_agents(d: DataView, a: TransferToHumanAgents) -> str:
    return "Transfer successful"


def cancel_pending_order(d: DataView, a: CancelPendingOrder) -> str:
    if a.order_id not in d.orders:
        return "Error: order not found"
    order = d.orders[a.order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be cancelled"
    refunds = []
    for payment in order["payment_history"]:
        pid = payment["payment_method_id"]
        refunds.append({"transaction_type": "refund", "amount": payment["amount"], "payment_method_id": pid})
        if "gift_card" in pid:
            pm = d.users[order["user_id"]]["payment_methods"][pid]
            pm["balance"] = round(pm["balance"] + payment["amount"], 2)
    order["status"] = "cancelled"
    order["cancel_reason"] = a.reason
    order["payment_history"].extend(refunds)
    return json.dumps(order)


def _address(a: ModifyPendingOrderAddress | ModifyUserAddress) -> dict[str, str]:
    return {
        "address1": a.address1,
        "address2": a.address2,
        "city": a.city,
        "state": a.state,
        "country": a.country,
        "zip": a.zip,
    }


def modify_pending_order_address(d: DataView, a: ModifyPendingOrderAddress) -> str:
    if a.order_id not in d.orders:
        return "Error: order not found"
    order = d.orders[a.order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"
    order["address"] = _address(a)
    return json.dumps(order)


def modify_pending_order_payment(d: DataView, a: ModifyPendingOrderPayment) -> str:
    if a.order_id not in d.orders:
        return "Error: order not found"
    order = d.orders[a.order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"
    methods = d.users[order["user_id"]]["payment_methods"]
    if a.payment_method_id not in methods:
        return "Error: payment method not found"
    history = order["payment_history"]
    if len(history) > 1 or history[0]["transaction_type"] != "payment":
        return "Error: there should be exactly one payment for a pending order"
    if history[0]["payment_method_id"] == a.payment_method_id:
        return "Error: the new payment method should be different from the current one"
    amount = history[0]["amount"]
    pm = methods[a.payment_method_id]
    if pm["source"] == "gift_card" and pm["balance"] < amount:
        return "Error: insufficient gift card balance to pay for the order"
    history.extend(
        [
            {"transaction_type": "payment", "amount": amount, "payment_method_id": a.payment_method_id},
            {"transaction_type": "refund", "amount": amount, "payment_method_id": history[0]["payment_method_id"]},
        ]
    )
    if pm["source"] == "gift_card":
        pm["balance"] = round(pm["balance"] - amount, 2)
    if "gift_card" in history[0]["payment_method_id"]:
        old = methods[history[0]["payment_method_id"]]
        old["balance"] = round(old["balance"] + amount, 2)
    return json.dumps(order)


def _price_diff(d: DataView, order: dict, item_ids: list[str], new_item_ids: list[str]) -> float | str:
    all_ids = [i["item_id"] for i in order["items"]]
    for iid in item_ids:
        if item_ids.count(iid) > all_ids.count(iid):
            return f"Error: {iid} not found"
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"
    diff = 0
    for iid, nid in zip(item_ids, new_item_ids, strict=True):
        item = next(i for i in order["items"] if i["item_id"] == iid)
        variants = d.products[item["product_id"]]["variants"]
        if not (nid in variants and variants[nid]["available"]):
            return f"Error: new item {nid} not found or available"
        diff += variants[nid]["price"] - item["price"]
    return diff


def modify_pending_order_items(d: DataView, a: ModifyPendingOrderItems) -> str:
    if a.order_id not in d.orders:
        return "Error: order not found"
    order = d.orders[a.order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"
    diff = _price_diff(d, order, a.item_ids, a.new_item_ids)
    if isinstance(diff, str):
        return diff
    methods = d.users[order["user_id"]]["payment_methods"]
    if a.payment_method_id not in methods:
        return "Error: payment method not found"
    pm = methods[a.payment_method_id]
    if pm["source"] == "gift_card" and pm["balance"] < diff:
        return "Error: insufficient gift card balance to pay for the new item"
    order["payment_history"].append(
        {
            "transaction_type": "payment" if diff > 0 else "refund",
            "amount": abs(diff),
            "payment_method_id": a.payment_method_id,
        }
    )
    if pm["source"] == "gift_card":
        pm["balance"] = round(pm["balance"] - diff, 2)
    for iid, nid in zip(a.item_ids, a.new_item_ids, strict=True):
        item = next(i for i in order["items"] if i["item_id"] == iid)
        variant = d.products[item["product_id"]]["variants"][nid]
        item["item_id"] = nid
        item["price"] = variant["price"]
        item["options"] = variant["options"]
    order["status"] = "pending (item modified)"
    return json.dumps(order)


def modify_user_address(d: DataView, a: ModifyUserAddress) -> str:
    if a.user_id not in d.users:
        return "Error: user not found"
    user = d.users[a.user_id]
    user["address"] = _address(a)
    return json.dumps(user)


def return_delivered_order_items(d: DataView, a: ReturnDeliveredOrderItems) -> str:
    if a.order_id not in d.orders:
        return "Error: order not found"
    order = d.orders[a.order_id]
    if order["status"] != "delivered":
        return "Error: non-delivered order cannot be returned"
    if a.payment_method_id not in d.users[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"
    if (
        "gift_card" not in a.payment_method_id
        and a.payment_method_id != order["payment_history"][0]["payment_method_id"]
    ):
        return "Error: payment method should be either the original payment method or a gift card"
    all_ids = [i["item_id"] for i in order["items"]]
    for iid in a.item_ids:
        if a.item_ids.count(iid) > all_ids.count(iid):
            return "Error: some item not found"
    order["status"] = "return requested"
    order["return_items"] = sorted(a.item_ids)
    order["return_payment_method_id"] = a.payment_method_id
    return json.dumps(order)


def exchange_delivered_order_items(d: DataView, a: ExchangeDeliveredOrderItems) -> str:
    if a.order_id not in d.orders:
        return "Error: order not found"
    order = d.orders[a.order_id]
    if order["status"] != "delivered":
        return "Error: non-delivered order cannot be exchanged"
    diff = _price_diff(d, order, a.item_ids, a.new_item_ids)
    if isinstance(diff, str):
        return diff
    diff = round(diff, 2)
    methods = d.users[order["user_id"]]["payment_methods"]
    if a.payment_method_id not in methods:
        return "Error: payment method not found"
    pm = methods[a.payment_method_id]
    if pm["source"] == "gift_card" and pm["balance"] < diff:
        return "Error: insufficient gift card balance to pay for the price difference"
    order["status"] = "exchange requested"
    order["exchange_items"] = sorted(a.item_ids)
    order["exchange_new_items"] = sorted(a.new_item_ids)
    order["exchange_payment_method_id"] = a.payment_method_id
    order["exchange_price_difference"] = diff
    return json.dumps(order)


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class ToolDef:
    name: str
    args: type[_Args]
    fn: Callable[[DataView, Any], str]
    kind: Kind

    def spec(self) -> dict[str, Any]:
        """Anthropic tool definition generated from the pydantic model."""
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("properties", {})
        return {"name": self.name, "description": self.args.__doc__ or "", "input_schema": schema}


def _t(name: str, args: type[_Args], fn: Callable, kind: Kind) -> ToolDef:
    return ToolDef(name, args, fn, kind)


TOOLS: dict[str, ToolDef] = {
    t.name: t
    for t in [
        _t("find_user_id_by_email", FindUserIdByEmail, find_user_id_by_email, Kind.READ),
        _t("find_user_id_by_name_zip", FindUserIdByNameZip, find_user_id_by_name_zip, Kind.READ),
        _t("get_user_details", GetUserDetails, get_user_details, Kind.READ),
        _t("get_order_details", GetOrderDetails, get_order_details, Kind.READ),
        _t("get_product_details", GetProductDetails, get_product_details, Kind.READ),
        _t("list_all_product_types", ListAllProductTypes, list_all_product_types, Kind.READ),
        _t("calculate", Calculate, calculate, Kind.META),
        _t("think", Think, think, Kind.META),
        _t("cancel_pending_order", CancelPendingOrder, cancel_pending_order, Kind.WRITE),
        _t("modify_pending_order_address", ModifyPendingOrderAddress, modify_pending_order_address, Kind.WRITE),
        _t("modify_pending_order_payment", ModifyPendingOrderPayment, modify_pending_order_payment, Kind.WRITE),
        _t("modify_pending_order_items", ModifyPendingOrderItems, modify_pending_order_items, Kind.WRITE),
        _t("modify_user_address", ModifyUserAddress, modify_user_address, Kind.WRITE),
        _t("return_delivered_order_items", ReturnDeliveredOrderItems, return_delivered_order_items, Kind.WRITE),
        _t("exchange_delivered_order_items", ExchangeDeliveredOrderItems, exchange_delivered_order_items, Kind.WRITE),
        _t("transfer_to_human_agents", TransferToHumanAgents, transfer_to_human_agents, Kind.HANDOFF),
    ]
}


def validate_args(name: str, raw: dict[str, Any]) -> _Args | str:
    """Parse model-supplied arguments; return an error string the model can act on."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool {name!r}"
    try:
        return tool.args.model_validate(raw)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'args'}: {err['msg']}" for err in e.errors())
        return f"Error: invalid arguments for {name}: {problems}"


def run_tool(data: DataView, name: str, args: _Args) -> str:
    return TOOLS[name].fn(data, args)


def refund_amount(data: DataView, name: str, args: _Args) -> float:
    """Money a write would send back to the customer (0 for non-refund actions)."""
    if name == "cancel_pending_order" and args.order_id in data.orders:
        order = data.orders[args.order_id]
        return round(sum(p["amount"] for p in order["payment_history"] if p["transaction_type"] == "payment"), 2)
    if name == "return_delivered_order_items" and args.order_id in data.orders:
        order = data.orders[args.order_id]
        remaining = [i for i in order["items"]]
        total = 0.0
        for iid in args.item_ids:
            match = next((i for i in remaining if i["item_id"] == iid), None)
            if match:
                total += match["price"]
                remaining.remove(match)
        return round(total, 2)
    return 0.0
