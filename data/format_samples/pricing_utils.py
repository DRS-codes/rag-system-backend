"""Small utility module — sample file for testing the code loader."""


def calculate_discount(price: float, percent_off: float) -> float:
    """Returns the price after applying a percentage discount."""
    if not 0 <= percent_off <= 100:
        raise ValueError("percent_off must be between 0 and 100")
    return round(price * (1 - percent_off / 100), 2)


def apply_bulk_pricing(unit_price: float, quantity: int) -> float:
    """
    Bulk pricing tiers:
    - 1-9 units: no discount
    - 10-49 units: 10% off
    - 50+ units: 20% off
    """
    if quantity >= 50:
        return calculate_discount(unit_price, 20) * quantity
    if quantity >= 10:
        return calculate_discount(unit_price, 10) * quantity
    return unit_price * quantity


class ShoppingCart:
    def __init__(self):
        self.items: list[tuple[str, float, int]] = []

    def add_item(self, name: str, unit_price: float, quantity: int) -> None:
        self.items.append((name, unit_price, quantity))

    def total(self) -> float:
        return sum(apply_bulk_pricing(price, qty) for _, price, qty in self.items)
