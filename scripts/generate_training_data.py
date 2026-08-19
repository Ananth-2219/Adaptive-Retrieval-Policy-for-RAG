import csv,os,re,collections,random

templates = {
    "factual": [
        "What is the capital of {c}?",
        "Who discovered {p}?",
        "When did {e} happen?",
        "What is the {a} {n} of {person}?",
        "Which {r} leads {o}?"
    ],
    "comparative": [
        "Is {x} bigger than {y}?",
        "Which is more popular: {x} or {y}?",
        "How does the {m} of {x} compare to that of {y}?",
        "Is {x} considered more {a} than {y} in {f}?",
        "Compare the {attr} of {x} and {y}."
    ],
    "definitional": [
        "What does {t} mean?",
        "Define {t}.",
        "What is the definition of {p}?",
        "What is the significance of {t} in {f}?",
        "How would you define {t} in everyday language?"
    ],
    "multihop": [
        "What {attr} did the {r} of {e} have before {e}?",
        "In what year was the {cr} of {w} born?",
        "How old was {person} when they {ach}?",
        "If {e1} led to {e2} and {e2} caused {e3}, what is the relationship between {e1} and {e3}?",
        "Given that {ent1} is {rel} to {ent2} and {ent2} is {rel} to {ent3}, what about {ent1} relative to {ent3}?"
    ]
}

entities = {
    "c": ["France","Japan","Brazil","Canada","India","Australia"],
    "p": ["gravity","photosynthesis","radioactivity","diffraction","evaporation","condensation"],
    "e": ["World War II","the Moon landing","the fall of the Berlin Wall","the invention of the telephone","the signing of the Magna Carta","the discovery of penicillin"],
    "a": ["major","significant","key","primary","central","essential"],
    "n": ["discovery","innovation","breakthrough","finding","result","event"],
    "person": ["Alice","Bob","Charlie","Diana","Eve","Frank"],
    "r": ["scientist","inventor","author","philosopher","leader","engineer"],
    "o": ["UN","NASA","World Health Organization","International Monetary Fund","European Commission","Red Cross"],
    "x": ["France","Germany","Japan","Brazil","Canada","Australia"],
    "y": ["Australia","China","Russia","South Africa","India","Italy"],
    "m": ["GDP","population","area","life expectancy","literacy rate","employment rate"],
    "f": ["biology","physics","economics","history","literature","psychology"],
    "t": ["algorithm","photosynthesis","blockchain","entropy","catalyst","neuron"],
    "attr": ["major achievement","key innovation","significant breakthrough","important discovery","critical development","transformative change"],
    "cr": ["Einstein","Da Vinci","Newton","Curie","Galileo","Heisenberg"],
    "w": ["Hamlet","The Theory of Relativity","On the Origin of Species","The Raven","The Great Gatsby","Moby Dick"],
    "ach": ["discovered penicillin","published the theory of relativity","invented the telephone","painted the Mona Lisa","decoded the DNA structure","created the first vaccine"],
    "e1": ["the Industrial Revolution","the invention of the telephone","World War II","the rise of factories","the discovery of penicillin","the launch of Sputnik"],
    "e2": ["the rise of factories","the discovery of penicillin","the fall of the Berlin Wall","the development of modern medicine","the invention of the internet","the Human Genome Project"],
    "e3": ["urbanization","the development of modern medicine","the spread of democracy","the invention of the internet","the Human Genome Project","the discovery of electricity"],
    "rel": ["greater than","leads to","part of","causes","results in","underlies"],
    "ent1": ["France","Germany","Japan","Brazil","Canada","Australia"],
    "ent2": ["Australia","China","Russia","South Africa","India","Italy"],
    "ent3": ["Italy","Spain","Portugal","Greece","Turkey","Brazil"]
}

def gen_intent(intent, needed=40):
    tmplist = templates[intent]
    examples = set()
    attempts = 0
    while len(examples) < needed and attempts < 5000:
        tmpl = random.choice(tmplist)
        ph = re.findall(r'\{(\w+)\}', tmpl)
        vals = {}
        for p in ph:
            vals[p] = random.choice(entities[p]) if p in entities else "X"
        try:
            sent = tmpl.format(**vals)
        except:
            attempts += 1
            continue
        if sent not in examples:
            examples.add(sent)
        attempts += 1
    if len(examples) < needed:
        i = 0
        while len(examples) < needed:
            for tmpl in tmplist:
                ph = re.findall(r'\{(\w+)\}', tmpl)
                vals = {p: entities[p][i % len(entities[p])] if p in entities else "X" for p in ph}
                try:
                    sent = tmpl.format(**vals)
                except:
                    continue
                if sent not in examples:
                    examples.add(sent)
                    if len(examples) >= needed:
                        break
            i += 1
    return list(examples)[:needed]

rows = []
for intent in ["factual","comparative","definitional","multihop"]:
    ex = gen_intent(intent, 40)
    for s in ex:
        rows.append({"query": s, "intent": intent})
    print(f"{intent}: generated {len(ex)} examples")

out_dir = "data/processed"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "intent_training_data.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["query","intent"])
    writer.writeheader()
    writer.writerows(rows)
print("Total rows:", len(rows))
cnt = collections.Counter(r["intent"] for r in rows)
for intent in ["factual","comparative","definitional","multihop"]:
    print(f"Final {intent} count: {cnt.get(intent,0)}")