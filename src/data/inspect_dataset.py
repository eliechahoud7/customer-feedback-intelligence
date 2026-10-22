from collections import Counter

from datasets import load_dataset


DATASET_NAME = "tomaarsen/setfit-absa-semeval-laptops"


def main():
    dataset = load_dataset(DATASET_NAME)

    train_data = dataset["train"]
    test_data = dataset["test"]

    print("=" * 60)
    print("TRAINING DATA ANALYSIS")
    print("=" * 60)

    train_texts = train_data["text"]

    print(f"Total training rows: {len(train_data)}")
    print(f"Unique training reviews: {len(set(train_texts))}")
    print(f"Repeated annotation rows: {len(train_data) - len(set(train_texts))}")

    print("\nTraining labels:")
    print(Counter(train_data["label"]))

    print("\n" + "=" * 60)
    print("TEST DATA ANALYSIS")
    print("=" * 60)

    test_texts = test_data["text"]

    print(f"Total test rows: {len(test_data)}")
    print(f"Unique test reviews: {len(set(test_texts))}")
    print(f"Rows with empty sentiment labels: {sum(not str(label).strip() for label in test_data['label'])}")

    overlap = set(train_texts) & set(test_texts)

    print(f"\nExact review texts shared by train and test: {len(overlap)}")

    print("\nExample of a review with multiple annotations:")

    example_text = train_texts[0]

    for row in train_data:
        if row["text"] == example_text:
            print({
                "text": row["text"],
                "aspect": row["span"],
                "sentiment": row["label"],
            })

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
