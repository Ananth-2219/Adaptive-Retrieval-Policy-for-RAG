import csv
import random
import os

templates = {
    "factual": [
        "What is the capital of {country}?",
        "Who discovered {phenomenon}?",
        "When did {event} happen?"
    ],
    "comparative": [
        "Is {X} bigger than {Y}?",
        "Which is more popular: {X} or {Y}?",
        "Compare the GDP of {X} and {Y}."
    ],
    "definitional": [
        "What does {term} mean?",
        "Define {concept}.",
        "What is the definition of {phenomenon}?"
    ],
    "multihop": [
        "If {person1} knows {person2}, and {person2} knows {person3}, does {person1} know {person3}?",
        "Given that {event1} led to {event2}, and {event2} caused {event3}, what is the relationship between {event1} and {event3}?",
        "If {cityA} is north of {cityB}, and {cityB} is north of {cityC}, where is {cityA} relative to {cityC}?"
    ]
}

entities = {
    "country": ["France", "Japan", "Brazil", "Canada", "India", "Australia", "Germany", "Nigeria", "Egypt", "Mexico", "Italy", "Spain", "South Korea", "Indonesia", "Turkey", "Netherlands", "Saudi Arabia", "Sweden", "Poland", "Belgium"],
    "phenomenon": ["gravity", "photosynthesis", "refraction", "evaporation", "magnetism", "radioactivity", "diffraction", "convection", "nuclear fission", "elasticity", "tornado", "hurricane", "earthquake", "volcano", "lightning", "aurora", "tsunami", "blizzard", "drought", "flood"],
    "event": ["World War II", "the Moon landing", "the fall of the Berlin Wall", "the invention of the telephone", "the signing of the Magna Carta", "the Industrial Revolution", "the discovery of penicillin", "the launch of Sputnik", "the Wright brothers' first flight", "the Panama Canal opening", "the fall of the Roman Empire", "the French Revolution", "the American Civil War", "the invention of the printing press", "the discovery of electricity", "the construction of the Great Wall", "the founding of the United Nations", "the invention of the internet", "the Human Genome Project", "the first Olympic Games"],
    "X": ["France", "Japan", "Brazil", "Canada", "India"],
    "Y": ["Germany", "Australia", "China", "Russia", "South Africa"],
    "term": ["photosynthesis", "blockchain", "artifact", "equilibrium", "algorithm", "entropy", "catalyst", "neuron", "genome", "isotope", "osmosis", "fermentation", "transpiration", "respiration", "germination", "pollination", "adaptation", "mutation", "homeostasis", "symbiosis"],
    "concept": ["democracy", "relativity", "ecosystem", "evolution", "quantum mechanics", "climate change", "artificial intelligence", "human rights", "supply and demand", "natural selection", "globalization", "urbanization", "industrialization", "modernization", "westernization", "democratization", "privatization", "liberalization", "centralization", "decentralization"],
    "person1": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "person2": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "person3": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "event1": ["the Industrial Revolution", "the invention of the steam engine", "the discovery of electricity", "the Wright brothers' flight", "the launch of Sputnik"],
    "event2": ["the rise of factories", "the spread of railways", "the telegraph", "the jet age", "the satellite era"],
    "event3": ["urbanization", "globalization", "the information age", "space exploration", "AI development"],
    "cityA": ["Tokyo", "Paris", "New York", "Sydney", "Mumbai"],
    "cityB": ["Tokyo", "Paris", "New York", "Sydney", "Mumbai"],
    "cityC": ["Tokyo", "Paris", "New York", "Sydney", "Mumbai"]
}

def generate_examples():
    random.seed(42)
    rows = []
    for intent, templist in templates.items():
        generated = set()
        attempts = 0
        while len(generated) < 40 and attempts < 200:
            attempts += 1
            tpl = random.choice(templist)
            import re
            places = re.findall(r'\{(.*?)\}', tpl)
            vals = {}
            for p in places:
                if p in entities:
                    vals[p] = random.choice(entities[p])
                else:
                    vals[p] = "X"
            try:
                sent = tpl.format(**vals)
            except KeyError:
                continue
            if sent not in generated:
                generated.add(sent)
                rows.append({"intent": intent, "query": sent})
        # If still not enough, fill with deterministic generation
        if len(generated) < 40:
            for tpl in templist:
                import re
                places = re.findall(r'\{(.*?)\}', tpl)
                # create all combos? simple: just fill with first entity each
                vals = {p: entities[p][0] if p in entities else "X" for p in places}
                try:
                    sent = tpl.format(**vals)
                except KeyError:
                    continue
                if sent not in generated:
                    generated.add(sent)
                    rows.append({"intent": intent, "query": sent})
                if len(generated) >= 40:
                    break
    return rows

def main():
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "intent_training_data.csv")
    rows = generate_examples()
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["intent", "query"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} examples saved to {out_path}")

if __name__ == "__main__":
    main()