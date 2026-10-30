from itg_llm.routing import ROUTES, route_for
from itg_llm.schemas import Category


def test_every_category_has_its_own_route():
    assert set(ROUTES) == set(Category)
    names = [route.name for route in ROUTES.values()]
    assert len(set(names)) == len(names)
    instructions = [route.instructions for route in ROUTES.values()]
    assert len(set(instructions)) == len(instructions)


def test_route_for():
    assert route_for(Category.SUPPORT) is ROUTES[Category.SUPPORT]


def test_only_support_asks_for_numbered_steps():
    numbered = [c for c, route in ROUTES.items() if route.numbered_steps]
    assert numbered == [Category.SUPPORT]


def test_sales_and_feedback_are_shorter_than_support_and_complaint():
    short = max(
        ROUTES[c].max_chars for c in (Category.SALES, Category.FEEDBACK)
    )
    long = min(
        ROUTES[c].max_chars for c in (Category.SUPPORT, Category.COMPLAINT)
    )
    assert short < long


def test_complaint_route_is_empathetic_and_support_is_technical():
    assert "frustration" in ROUTES[Category.COMPLAINT].instructions
    assert "numbered list" in ROUTES[Category.SUPPORT].instructions
    assert "Do not invent prices" in ROUTES[Category.SALES].instructions
