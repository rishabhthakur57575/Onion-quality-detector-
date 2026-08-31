from pathlib import Path

BASE_DIR = Path(__file__).parent

for split in ["train", "valid"]:

    image_dir = BASE_DIR / split / "images"
    label_dir = BASE_DIR / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split.upper()}")
    print("-" * 30)

    print("Images:", len(images))
    print("Labels:", len(labels))

    missing = 0

    for image in images:

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            print("Missing label:", image.name)
            missing += 1

    print("Missing labels:", missing)
