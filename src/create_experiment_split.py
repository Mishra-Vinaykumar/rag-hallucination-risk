import os
import random
import pandas as pd

from datasets import load_dataset


# ==========================================
# Configuration
# ==========================================

RANDOM_SEED = 42

DEVELOPMENT_SIZE = 100
TRAIN_SIZE = 2000
VALIDATION_SIZE = 500
FINAL_TEST_SIZE = 500


OUTPUT_DIR = (
    "data/processed/experiment_splits"
)


# ==========================================
# Create output directory
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# Load HotpotQA
# ==========================================

print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)


train_data = dataset["train"]
official_validation = dataset["validation"]


print(
    "Train examples:",
    len(train_data)
)

print(
    "Official validation examples:",
    len(official_validation)
)


# ==========================================
# Development set
# ==========================================
#
# IMPORTANT:
# These are the first 100 train examples
# already used during development.
#
# They must NEVER be used for final model
# evaluation.
# ==========================================

development_indices = list(
    range(DEVELOPMENT_SIZE)
)


development_data = train_data.select(
    development_indices
)


# ==========================================
# Remaining training pool
# ==========================================

remaining_train_indices = list(
    range(
        DEVELOPMENT_SIZE,
        len(train_data)
    )
)


rng = random.Random(
    RANDOM_SEED
)


rng.shuffle(
    remaining_train_indices
)


# ==========================================
# Training split
# ==========================================

training_indices = (
    remaining_train_indices[
        :TRAIN_SIZE
    ]
)


training_data = train_data.select(
    training_indices
)


# ==========================================
# Validation split
# ==========================================

validation_start = TRAIN_SIZE

validation_end = (
    TRAIN_SIZE
    +
    VALIDATION_SIZE
)


validation_indices = (
    remaining_train_indices[
        validation_start:
        validation_end
    ]
)


validation_data = train_data.select(
    validation_indices
)


# ==========================================
# Final test split
# ==========================================
#
# Use the official HotpotQA validation split
# as an untouched final holdout source.
# ==========================================

official_validation_indices = list(
    range(
        len(official_validation)
    )
)


rng_test = random.Random(
    RANDOM_SEED
)


rng_test.shuffle(
    official_validation_indices
)


final_test_indices = (
    official_validation_indices[
        :FINAL_TEST_SIZE
    ]
)


final_test_data = (
    official_validation.select(
        final_test_indices
    )
)


# ==========================================
# Helper function
# ==========================================

def create_split_dataframe(
    split_data,
    experiment_split,
    source_split
):

    rows = []


    for record in split_data:

        row = {

            "query_id":
                record["id"],

            "question":
                record["question"],

            "answer":
                record["answer"],

            "question_type":
                record["type"],

            "difficulty":
                record["level"],

            "experiment_split":
                experiment_split,

            "source_split":
                source_split

        }


        rows.append(
            row
        )


    return pd.DataFrame(
        rows
    )


# ==========================================
# Create DataFrames
# ==========================================

development_df = (
    create_split_dataframe(
        development_data,
        "development",
        "train"
    )
)


training_df = (
    create_split_dataframe(
        training_data,
        "training",
        "train"
    )
)


validation_df = (
    create_split_dataframe(
        validation_data,
        "validation",
        "train"
    )
)


final_test_df = (
    create_split_dataframe(
        final_test_data,
        "final_test",
        "official_validation"
    )
)


# ==========================================
# Check for query leakage
# ==========================================

development_ids = set(
    development_df["query_id"]
)

training_ids = set(
    training_df["query_id"]
)

validation_ids = set(
    validation_df["query_id"]
)

final_test_ids = set(
    final_test_df["query_id"]
)


assert development_ids.isdisjoint(
    training_ids
)

assert development_ids.isdisjoint(
    validation_ids
)

assert development_ids.isdisjoint(
    final_test_ids
)

assert training_ids.isdisjoint(
    validation_ids
)

assert training_ids.isdisjoint(
    final_test_ids
)

assert validation_ids.isdisjoint(
    final_test_ids
)


print(
    "\nNo query leakage detected!"
)


# ==========================================
# Save individual split files
# ==========================================

development_path = os.path.join(
    OUTPUT_DIR,
    "development_queries.csv"
)


training_path = os.path.join(
    OUTPUT_DIR,
    "training_queries.csv"
)


validation_path = os.path.join(
    OUTPUT_DIR,
    "validation_queries.csv"
)


final_test_path = os.path.join(
    OUTPUT_DIR,
    "final_test_queries.csv"
)


development_df.to_csv(
    development_path,
    index=False
)


training_df.to_csv(
    training_path,
    index=False
)


validation_df.to_csv(
    validation_path,
    index=False
)


final_test_df.to_csv(
    final_test_path,
    index=False
)


# ==========================================
# Combined split manifest
# ==========================================

split_manifest = pd.concat(
    [
        development_df,
        training_df,
        validation_df,
        final_test_df
    ],
    ignore_index=True
)


manifest_path = os.path.join(
    OUTPUT_DIR,
    "experiment_split_manifest.csv"
)


split_manifest.to_csv(
    manifest_path,
    index=False
)


# ==========================================
# Summary
# ==========================================

print("\n")
print("=" * 70)
print("EXPERIMENT SPLIT COMPLETE")
print("=" * 70)


print(
    "\nDevelopment questions:"
)

print(
    len(development_df)
)


print(
    "\nTraining questions:"
)

print(
    len(training_df)
)


print(
    "\nValidation questions:"
)

print(
    len(validation_df)
)


print(
    "\nFinal test questions:"
)

print(
    len(final_test_df)
)


print(
    "\nTotal unique questions:"
)

print(
    len(split_manifest)
)


print(
    "\nUnique query IDs:"
)

print(
    split_manifest[
        "query_id"
    ].nunique()
)


print(
    "\nRandom seed:"
)

print(
    RANDOM_SEED
)


print(
    "\nFiles saved in:"
)

print(
    OUTPUT_DIR
)


print(
    "\nIMPORTANT:"
)

print(
    "The final_test split must remain "
    "untouched until final evaluation."
)