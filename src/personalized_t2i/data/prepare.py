import csv
from pathlib import Path

from PIL import Image, ImageOps


SELECTION_ORDER = {
    "cat_mug": [
        "img_7898",
        "img_7901",
        "img_7904",
        "img_7899",
        "img_7902",
        "img_7900",
        "img_7903",
        "img_7905",
        "img_7916",
        "img_7917",
    ],
    "dog_plush": [
        "img_7907",
        "img_7911",
        "img_7912",
        "img_7909",
        "img_7910",
        "img_7908",
        "img_7913",
        "img_7914",
        "img_7915",
        "img_7918",
    ],
    "blue_white_vase": [
        "img_7888",
        "img_7891",
        "img_7893",
        "img_7895",
        "img_7897",
        "img_7889",
        "img_7890",
        "img_7892",
        "img_7894",
        "img_7896",
    ],
}


def subset_membership(position: int) -> str:
    memberships = []

    for size in (1, 3, 5, 10):
        if position <= size:
            memberships.append(str(size))

    return ",".join(memberships)


def assign_nested_subsets(manifest_path: Path) -> None:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    concept_id = next(
        row["concept_id"] for row in rows if row["split"] == "train_pool"
    )

    order = SELECTION_ORDER[concept_id]

    if len(order) != 10 or len(set(order)) != 10:
        raise ValueError(
            f"{concept_id}: selection order must contain 10 unique images"
        )

    positions = {
        image_id: index + 1
        for index, image_id in enumerate(order)
    }

    train_ids = {
        row["image_id"]
        for row in rows
        if row["split"] == "train_pool"
    }

    if train_ids != set(order):
        raise ValueError(
            f"{concept_id}: selection order does not match train_pool"
        )

    for row in rows:
        if row["split"] == "train_pool":
            row["subset_membership"] = subset_membership(
                positions[row["image_id"]]
            )
        else:
            row["subset_membership"] = ""

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def verify_nested_subsets(manifest_path: Path) -> None:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row["split"] == "train_pool"
        ]

    concept_id = rows[0]["concept_id"]

    subsets = {}

    for size in (1, 3, 5, 10):
        subsets[size] = {
            row["image_id"]
            for row in rows
            if str(size) in row["subset_membership"].split(",")
        }

        if len(subsets[size]) != size:
            raise ValueError(
                f"{concept_id}: D{size} contains "
                f"{len(subsets[size])} images"
            )

    if not (
        subsets[1] < subsets[3]
        and subsets[3] < subsets[5]
        and subsets[5] < subsets[10]
    ):
        raise ValueError(
            f"{concept_id}: nested subset check failed"
        )

    print(
        f"{concept_id}: "
        "D1=1 D3=3 D5=5 D10=10 PASS"
    )


def preprocess_image(
    input_path: Path,
    output_path: Path,
) -> None:
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        if image.width != image.height:
            raise ValueError(
                f"source image is not square: "
                f"{input_path} {image.size}"
            )

        image = image.resize(
            (512, 512),
            Image.Resampling.LANCZOS,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(output_path, quality=95)


def process_manifest(manifest_path: Path) -> int:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    concept_id = next(
        row["concept_id"] for row in rows
        if row["split"] == "train_pool"
    )

    output_dir = (
        Path("data/processed")
        / concept_id
        / "v1"
        / "train_pool"
    )

    temp_dir = output_dir.with_name("train_pool_tmp")

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in temp_dir.glob("*.jpeg"):
        old_file.unlink()

    processed = 0

    for row in rows:
        if row["split"] != "train_pool":
            continue

        input_path = Path(row["file_path"])

        output_path = (
            temp_dir
            / input_path.with_suffix(".jpeg").name
        )

        preprocess_image(
            input_path,
            output_path,
        )

        processed += 1

    if processed != 10:
        raise ValueError(
            f"{concept_id}: expected 10 processed images, "
            f"processed {processed}"
        )

    temp_files = sorted(temp_dir.glob("*.jpeg"))

    if len(temp_files) != 10:
        raise ValueError(
            f"{concept_id}: expected 10 temporary images, "
            f"found {len(temp_files)}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in output_dir.glob("*.jpeg"):
        old_file.unlink()

    for temp_file in temp_files:
        temp_file.replace(output_dir / temp_file.name)

    temp_dir.rmdir()

    return processed


def verify_processed() -> None:
    total = 0

    for concept_id in SELECTION_ORDER:
        folder = (
            Path("data/processed")
            / concept_id
            / "v1"
            / "train_pool"
        )

        files = sorted(folder.glob("*.jpeg"))

        if len(files) != 10:
            raise ValueError(
                f"{concept_id}: expected 10 processed images, "
                f"found {len(files)}"
            )

        for path in files:
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise ValueError(
                        f"invalid size: {path} {image.size}"
                    )

                if image.mode != "RGB":
                    raise ValueError(
                        f"invalid mode: {path} {image.mode}"
                    )

        total += len(files)
        print(f"{concept_id}: 10/10 PASS")

    print(f"processed total: {total}/30")
    print("512x512 RGB: PASS")


def main() -> None:
    manifests = [
        Path("data/manifests/cat_mug_v1.csv"),
        Path("data/manifests/dog_plush_v1.csv"),
        Path("data/manifests/blue_white_vase_v1.csv"),
    ]

    for manifest in manifests:
        assign_nested_subsets(manifest)
        verify_nested_subsets(manifest)

    total = 0

    for manifest in manifests:
        total += process_manifest(manifest)

    print(f"preprocessed: {total} images")

    verify_processed()


if __name__ == "__main__":
    main()
