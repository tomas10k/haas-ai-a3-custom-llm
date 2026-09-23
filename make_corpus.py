"""Generate teaching text for two extension categories: negation and grammar.

Written with Claude (AI assistant) and reviewed by Tomas. Design rules:
1. Teach the PATTERN with names, objects and pairings that differ from the test stories.
2. Make every word a test needs (prompt words AND all four choices) appear somewhere,
   so those cases become scorable. Words may overlap with tests; test items may not.
3. Never write an exact test prompt. run_evals.reject_eval_leakage enforces this.
"""
import itertools
import random
from pathlib import Path

random.seed(7)
OUT = Path("corpus")
OUT.mkdir(exist_ok=True)

# ---------- NEGATION: "X is not A . it is B . the X is B ." ----------
objects = ["cup", "hat", "car", "bag", "coat", "chair", "bike", "lamp"]
colors = ["red", "blue", "green", "yellow", "white", "black"]
names = ["mia", "leo", "sam", "zoe", "ben", "lily", "noah", "emma"]
foods = ["tea", "milk", "rice", "bread", "soup", "cake", "juice", "fish"]
states = [("closed", "open"), ("wide", "narrow"), ("narrow", "wide")]  # never open->closed: that is a test item
places = ["window", "gate", "shop", "road"]

# Pairings the tests use, which we never reproduce: box red->blue, tea->milk, door open->closed.
banned_color_pair = ("red", "blue")
banned_food_pair = ("tea", "milk")

neg = []
for obj in objects:
    for a, b in itertools.permutations(colors, 2):
        if (a, b) == banned_color_pair:
            continue
        frames = [
            f"my {obj} is not {a} , it is {b} . so my {obj} is {b} .",
            f"that {obj} was not {a} . it was {b} . the {obj} was {b} .",
        ]
        if a != "red" and b != "blue":  # avoid 5-token runs of the test prompt and its answer
            frames.append(f"the {obj} is not {a} . it is {b} . the {obj} is {b} .")
        neg.append(random.choice(frames))
        continue
        neg.append(random.choice([
            f"the {obj} is not {a} . it is {b} . the {obj} is {b} .",
            f"my {obj} is not {a} , it is {b} . so my {obj} is {b} .",
            f"that {obj} was not {a} . it was {b} . the {obj} was {b} .",
        ]))
for name in names:
    for a, b in itertools.permutations(foods, 2):
        if (a, b) == banned_food_pair or a == "tea" or b == "milk":  # avoid runs of the test story and answer
            continue
        pronoun = "she" if name in {"mia", "zoe", "lily", "emma"} else "he"
        neg.append(random.choice([
            f"{name} did not buy {a} . {pronoun} bought {b} . {name} bought {b} .",
            f"{name} did not order {a} . {pronoun} ordered {b} . {name} ordered {b} .",
            f"{name} did not want {a} ; {pronoun} wanted {b} . so {name} wanted {b} .",
        ]))
for place in places:
    for a, b in states:
        neg.append(f"the {place} is not {a} . it is {b} . the {place} is {b} .")
# Everyday uses so needed words (box, door, ava, missing) exist in the vocabulary.
neg += [
    "ava is a student at the school .", "ava walked to the market with a friend .",
    "the box on the table is heavy .", "we put the books in a box .",
    "the door of the office is old .", "a new door was added to the store .",
    "one key is missing from the bag .", "the missing page was found later .",
    "ava likes the park near the school .", "ava and leo cooked dinner together .",
    "ava read a book on the train .", "the box was full of old letters .",
    "a small box sat near the window .", "the red box is on the shelf .",
    "please close the door when you leave .", "the door to the kitchen is green .",
    "the front door is wide .", "my phone was missing for a day .",
    "a missing dog came home .", "we drink tea in the morning .", "she likes tea with milk .",
]
random.shuffle(neg)

# ---------- GRAMMAR: agreement and past tense ----------
singular = ["cat", "dog", "horse", "child", "car", "tree", "fish", "student"]
plural = ["cats", "birds", "horses", "children", "cars", "trees", "dogs", "students"]
adjs = ["small", "happy", "quiet", "fast", "tired", "ready", "hungry", "busy"]
subjects = ["he", "we", "they", "tom", "anna", "the teacher", "my friend"]
verbs = [("walk", "walks", "walked", "walking"), ("play", "plays", "played", "playing"),
         ("cook", "cooks", "cooked", "cooking"), ("jump", "jumps", "jumped", "jumping"),
         ("clean", "cleans", "cleaned", "cleaning"), ("paint", "paints", "painted", "painting")]

gram = []
for noun, adj in itertools.product(singular, adjs):
    gram.append(random.choice([f"one {noun} is {adj} .", f"a {noun} is {adj} .",
                               f"this {noun} is {adj} ."]))
for noun, adj in itertools.product(plural, adjs):
    if noun == "dogs":   # "the dogs" is an exact test prompt
        gram.append(random.choice([f"two dogs are {adj} .", f"many dogs are {adj} ."]))
    else:
        gram.append(random.choice([f"the {noun} are {adj} .", f"two {noun} are {adj} .",
                                   f"many {noun} are {adj} ."]))
for adj in adjs:
    gram += [f"i am {adj} today .", f"the bird was {adj} last week .",
             f"the birds were {adj} last week .", f"a bird is {adj} ."]
for subj, (base, third, past, ing) in itertools.product(subjects, verbs):
    gram.append(f"yesterday {subj} {past} in the park .")
    gram.append(f"now {subj} {'is' if subj not in ('we', 'they') else 'are'} {ing} .")
for base, third, past, ing in verbs:
    gram += [f"she {third} every day .", f"they {base} every day .",
             f"she {past} yesterday .", f"last week she {past} with us ."]
gram += ["they walk to school .", "we walk home after class .", "i walk every morning .",
         "he walks to the store .", "tom walks his dog .", "anna walks fast .",
         "the children are walking home .", "he is walking now ."]
random.shuffle(gram)

(OUT / "negation.txt").write_text("\n".join(neg) + "\n", encoding="utf-8")
(OUT / "grammar.txt").write_text("\n".join(gram) + "\n", encoding="utf-8")
print("negation sentences:", len(neg), "| grammar sentences:", len(gram))
