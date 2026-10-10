"""Fixed four-shot prompts and strict final boxed-answer scoring."""
import re
from decimal import Decimal, InvalidOperation

COT_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_FEWSHOT = [
    {
        "question": (
            "Angelo and Melanie want to plan how many hours over the next week they "
            "should study together for their test next week. They have 2 chapters of "
            "their textbook to study and 4 worksheets to memorize. They figure out "
            "that they should dedicate 3 hours to each chapter of their textbook and "
            "1.5 hours for each worksheet. If they plan to study no more than 4 hours "
            "each day, how many days should they plan to study total over the next "
            "week if they take a 10-minute break every hour, include 3 10-minute "
            "snack breaks each day, and 30 minutes for lunch each day?"
        ),
        "answer": (
            "Angelo and Melanie think they should dedicate 3 hours to each of the 2 "
            "chapters, 3 hours x 2 chapters = 6 hours total.\n"
            "For the worksheets they plan to dedicate 1.5 hours for each worksheet, "
            "1.5 hours x 4 worksheets = 6 hours total.\n"
            "Angelo and Melanie need to start with planning 12 hours to study, at 4 "
            "hours a day, 12 / 4 = 3 days.\n"
            "However, they need to include time for breaks and lunch. Every hour they "
            "want to include a 10-minute break, so 12 total hours x 10 minutes = 120 "
            "extra minutes for breaks.\n"
            "They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 "
            "minutes.\n"
            "And they want to include 30 minutes for lunch each day, so 120 minutes "
            "for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 "
            "minutes, or 180 / 60 minutes per hour = 3 extra hours.\n"
            "So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks "
            "= 15 hours total.\n"
            "They want to study no more than 4 hours each day, 15 hours / 4 hours "
            "each day = 3.75\n"
            "They will need to plan to study 4 days to allow for all the time they "
            "need.\n"
            "The answer is $\\boxed{4}$."
        ),
    },
    {
        "question": (
            "Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free "
            "throws. Their opponents score double the 2 pointers but half the 3 "
            "pointers and free throws. What's the total number of points scored by "
            "both teams added together?"
        ),
        "answer": (
            "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in "
            "2 pointers.\n"
            "His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in "
            "3 pointers\n"
            "They scored 10 free throws, and free throws count as one point so they "
            "scored 10*1=10 points in free throws.\n"
            "All together his team scored 50+24+10= 84 points\n"
            "Mark's opponents scored double his team's number of 2 pointers, meaning "
            "they scored 50*2=100 points in 2 pointers.\n"
            "His opponents scored half his team's number of 3 pointers, meaning they "
            "scored 24/2= 12 points in 3 pointers.\n"
            "They also scored half Mark's team's points in free throws, meaning they "
            "scored 10/2=5 points in free throws.\n"
            "All together Mark's opponents scored 100+12+5=117 points\n"
            "The total score for the game is both team's scores added together, so it "
            "is 84+117=201 points\n"
            "The answer is $\\boxed{201}$."
        ),
    },
    {
        "question": (
            "Bella has two times as many marbles as frisbees. She also has 20 more "
            "frisbees than deck cards. If she buys 2/5 times more of each item, what "
            "would be the total number of the items she will have if she currently has "
            "60 marbles?"
        ),
        "answer": (
            "When Bella buys 2/5 times more marbles, she'll have increased the number "
            "of marbles by 2/5*60 = 24\n"
            "The total number of marbles she'll have is 60+24 = 84\n"
            "If Bella currently has 60 marbles, and she has two times as many marbles "
            "as frisbees, she has 60/2 = 30 frisbees.\n"
            "If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more "
            "frisbees.\n"
            "The total number of frisbees she'll have will increase to 30+12 = 42\n"
            "Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = "
            "10 deck cards\n"
            "If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck "
            "cards.\n"
            "The total number of deck cards she'll have is 10+4 = 14\n"
            "Together, Bella will have a total of 14+42+84 = 140 items\n"
            "The answer is $\\boxed{140}$."
        ),
    },
    {
        "question": (
            "A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas "
            "in the first three baskets and 2 less of each fruit in the fourth basket. "
            "How many fruits are there?"
        ),
        "answer": (
            "For the first three baskets, the number of apples and oranges in one "
            "basket is 9+15=24\n"
            "In total, together with bananas, the number of fruits in one basket is "
            "24+14=38 for the first three baskets.\n"
            "Since there are three baskets each having 38 fruits, there are 3*38=114 "
            "fruits in the first three baskets.\n"
            "The number of apples in the fourth basket is 9-2=7\n"
            "There are also 15-2=13 oranges in the fourth basket\n"
            "The combined number of oranges and apples in the fourth basket is 13+7=20\n"
            "The fourth basket also contains 14-2=12 bananas.\n"
            "In total, the fourth basket has 20+12=32 fruits.\n"
            "The four baskets together have 32+114=146 fruits.\n"
            "The answer is $\\boxed{146}$."
        ),
    },
]


def fewshot_samples():
    return _FEWSHOT


def doc_to_text(doc):
    return f"Question: {doc['question']}\n{COT_INSTRUCTION}"


def doc_to_target(doc):
    return doc["answer"]


def number(text):
    """Accept one complete number, with optional grouping commas or dollar sign."""
    text = text.strip().removeprefix("$").strip()
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        return None
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def extract_prediction(text):
    # Fail the run instead of silently scoring malformed tokenizer output.
    if any(marker in text for marker in ("Ġ", "Ċ", "ĉ")):
        raise ValueError("Undecoded byte-level token markers: repair the tokenizer decoder and rerun.")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None  # unfinished reasoning is not a final answer
    start = text.rfind(r"\boxed")
    if start < 0:
        return None
    # No fallback to intermediate 'total is' phrases or arbitrary last numbers.
    match = re.match(r"\\boxed\s*\{([^{}]*)\}", text[start:])
    return number(match.group(1)) if match else None


def process_results(doc, results):
    prediction = extract_prediction(results[0])
    gold = number(doc["answer"].rsplit("####", 1)[-1])
    if gold is None:
        raise ValueError("Invalid GSM8K target")
    return {"exact_match": float(prediction is not None and prediction == gold),
            "boxed_rate": float(prediction is not None)}
