import os
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from opik.integrations.genai import track_genai
load_dotenv()

# Load your training data
df = pd.read_csv("data/train.csv")

# Prompt template
PROMPT_TEMPLATE = """You are a content moderation assistant.
Your task is to determine whether a given comment violates a specific rule.

### Rule:
{rule}

### Examples of Violations:
1. {pos1}
2. {pos2}

### Examples of Non-Violations:
1. {neg1}
2. {neg2}

### Comment to Evaluate:
{body}

### Task:
Does the above comment violate the rule?
Answer only with:
- "1" if it violates the rule
- "0" if it does not violate the rule
"""


def make_prompt(row):
    return PROMPT_TEMPLATE.format(
        rule=row["rule"],
        pos1=row["positive_example_1"],
        pos2=row["positive_example_2"],
        neg1=row["negative_example_1"],
        neg2=row["negative_example_2"],
        body=row["body"]
    )


def ask_gemini(prompt, client, model="gemini-2.5-flash-lite"):
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    config = types.GenerateContentConfig(
        max_output_tokens=1,  # short outputs
        response_mime_type="text/plain",
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return response.candidates[0].content.parts[0].text.strip()


def main():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    client = track_genai(client)

    predictions = []
    for i, row in df.head(5).iterrows():  # run only first 5 rows for testing
        prompt = make_prompt(row)
        prediction = ask_gemini(prompt, client)
        predictions.append(prediction)
        print(f"Comment: {row['body'][:60]}...")
        print(f"Prediction: {prediction}\n")

    # Save results
    df_subset = df.head(5).copy()
    df_subset["prediction"] = predictions
    df_subset.to_csv("predictions.csv", index=False)
    print("✅ Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()

