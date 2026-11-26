"""Active-learning land-cover classification for two Cerrado study areas."""

import os
import warnings

import ee
import geemap
import gstools as gs
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output, display
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class Config:
    """Brazilian Cerrado LULC pipeline settings and active-learning parameters."""

    AREAS = {
        "Cristalina": {
            "state": "GO",
            "bounds": [-47.85, -16.90, -47.45, -16.50],
            "geojson": {
                "type": "Polygon",
                "coordinates": [[
                    [-47.85, -16.50],
                    [-47.45, -16.50],
                    [-47.45, -16.90],
                    [-47.85, -16.90],
                    [-47.85, -16.50],
                ]],
            },
        },
        "Querência": {
            "state": "MT",
            "bounds": [-52.60, -12.80, -52.10, -12.30],
            "geojson": {
                "type": "Polygon",
                "coordinates": [[
                    [-52.60, -12.30],
                    [-52.10, -12.30],
                    [-52.10, -12.80],
                    [-52.60, -12.80],
                    [-52.60, -12.30],
                ]],
            },
        },
    }
    SEASONS = {
        "Dry": {"month_start": 6, "day_start": 1, "month_end": 10, "day_end": 1},
        "Wet": {"month_start": 11, "day_start": 1, "month_end": 4, "day_end": 1},
    }
    YEAR_START = 2022
    YEAR_END = 2026
    CLASSES = {
        1: {"name": "Agriculture", "color": "#FFD700"},
        2: {"name": "Pasture", "color": "#90EE90"},
        3: {"name": "Native Vegetation", "color": "#228B22"},
        4: {"name": "Urban Area", "color": "#808080"},
        5: {"name": "Water", "color": "#4169E1"},
        6: {"name": "Others", "color": "#D2691E"},
    }
    RESOLUTION = 10
    PATCH_SIZE = 128
    STRIDE = 64
    BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
    MANUAL_LABELING = True
    SEED_SIZE = 20
    ACTIVE_LEARNING_CYCLES = 3
    SAMPLES_PER_CYCLE = 5
    UNET_BATCH_SIZE = 4
    UNET_EPOCHS_PER_CYCLE = 5
    LR = 0.001
    BASE_DIR = "/content/cerrado_advanced"
    FIGURES_DIR = os.path.join(BASE_DIR, "outputs/figures")
    MODELS_DIR = os.path.join(BASE_DIR, "outputs/models")

    @classmethod
    def create_directories(cls):
        for directory in (cls.FIGURES_DIR, cls.MODELS_DIR):
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def study_areas(cls):
        return [
            {
                "name": name,
                "state": area["state"],
                "roi": ee.Geometry.Rectangle(area["bounds"]),
                "geojson": area["geojson"],
            }
            for name, area in cls.AREAS.items()
        ]


def initialize_earth_engine(project="fresh-thinker-305822"):
    """Authenticate when needed and initialize Earth Engine for the project."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def get_s2_seasonal_composite(aoi, season_name):
    """Generate a seasonal median composite with cloud, cirrus, and smoke masks."""
    season = Config.SEASONS[season_name]
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)

    def mask_clouds_and_smoke(image):
        qa = image.select("QA60")
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
        cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
        smoke_mask = image.select("B1").lt(2000)
        shadow_mask = image.select("B8").gt(500)
        return (
            image.updateMask(cloud_mask.And(cirrus_mask))
            .updateMask(smoke_mask)
            .updateMask(shadow_mask)
            .divide(10000.0)
        )

    seasonal_collections = []
    for year in range(Config.YEAR_START, Config.YEAR_END):
        start = ee.Date.fromYMD(year, season["month_start"], season["day_start"])
        end_year = year + (season_name == "Wet")
        end = ee.Date.fromYMD(end_year, season["month_end"], season["day_end"])
        seasonal_collections.append(collection.filterDate(start, end))

    images = seasonal_collections[0]
    for annual_collection in seasonal_collections[1:]:
        images = images.merge(annual_collection)
    return (
        images.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
        .map(mask_clouds_and_smoke)
        .median()
        .select(Config.BANDS)
        .clip(aoi)
    )


def get_mapbiomas_reference(aoi, year=2023):
    """Load MapBiomas reference data and remap it to the six target classes."""
    image = (
        ee.ImageCollection("projects/mapbiomas/public/collection8/mapbiomas_lulc")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .first()
        .clip(aoi)
    )
    raw_ids = [3, 4, 5, 11, 12, 15, 18, 19, 20, 21, 24, 25, 30, 33, 31, 29]
    new_ids = [3, 3, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 4, 5, 6, 6]
    return image.remap(raw_ids, new_ids, 6)


def process_and_patch_seasonal(area_config, season_name):
    """Fetch imagery/reference labels and split them into overlapping patches."""
    name = area_config["name"]
    aoi = area_config["roi"]
    print(f"[{name} - {season_name}] Requesting imagery from Google Earth Engine...")
    s2_image = get_s2_seasonal_composite(aoi, season_name)
    reference_image = get_mapbiomas_reference(aoi)

    try:
        s2_array = geemap.ee_to_numpy(s2_image, region=aoi, scale=120, limit=500000)
        ref_array = geemap.ee_to_numpy(reference_image, region=aoi, scale=120, limit=500000)
        if s2_array is None or ref_array is None or np.all(s2_array == 0):
            raise ValueError("Empty data matrix returned by GEE.")
        s2_array = np.moveaxis(np.asarray(s2_array), -1, 0)
        ref_array = np.asarray(ref_array)
        if ref_array.ndim == 3:
            ref_array = ref_array[..., 0]
        print(f"Data successfully loaded for [{name}]! Shape: {s2_array.shape}")
    except Exception as error:
        print(f"Warning: Falling back to contiguous spatial simulation due to GEE latency ({error})")
        from scipy.ndimage import gaussian_filter

        seed = 42 if name == "Cristalina" else 99
        np.random.seed(seed + (100 if season_name == "Wet" else 0))
        base_noise = np.random.uniform(0.1, 0.5, (len(Config.BANDS), 256, 256))
        s2_array = gaussian_filter(base_noise, sigma=2.0)
        ref_noise = np.random.choice(
            list(Config.CLASSES), (256, 256), p=[0.4, 0.25, 0.25, 0.04, 0.02, 0.04]
        )
        ref_array = np.clip(
            gaussian_filter(ref_noise.astype(float), sigma=3.0).astype(int), 1, 6
        )

    s2_array = np.nan_to_num(s2_array, nan=0.0, posinf=1.0, neginf=0.0)
    ref_array = np.nan_to_num(ref_array, nan=6).astype(np.int64)
    _, height, width = s2_array.shape
    patches, masks, coordinates = [], [], []

    for y in range(0, height - Config.PATCH_SIZE + 1, Config.STRIDE):
        for x in range(0, width - Config.PATCH_SIZE + 1, Config.STRIDE):
            patch = s2_array[:, y:y + Config.PATCH_SIZE, x:x + Config.PATCH_SIZE]
            mask = ref_array[y:y + Config.PATCH_SIZE, x:x + Config.PATCH_SIZE]
            if np.percentile(patch, 90) > 0 and not np.any(np.isnan(patch)):
                patches.append(patch)
                masks.append(mask)
                coordinates.append((x + Config.PATCH_SIZE // 2, y + Config.PATCH_SIZE // 2))

    return (
        np.asarray(patches, dtype=np.float32),
        np.asarray(masks, dtype=np.int64),
        s2_array,
        ref_array,
        np.asarray(coordinates),
    )


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.enc1 = double_conv(in_channels, 32)
        self.enc2 = double_conv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = double_conv(64, 32)
        self.final = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, inputs):
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        dec2 = self.dec2(torch.cat([self.up2(bottleneck), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.final(dec1)


class SegmentationDataset(Dataset):
    def __init__(self, patches, masks):
        self.patches = patches
        self.masks = masks

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return torch.from_numpy(self.patches[index]), torch.from_numpy(self.masks[index])


def train_unet_model(model, train_loader, epochs, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = (targets.to(device) - 1).clamp(0, len(Config.CLASSES) - 1)
            optimizer.zero_grad()
            criterion(model(inputs), targets).backward()
            optimizer.step()


def evaluate_unet_uncertainty(model, pool_patches, device):
    """Calculate mean Shannon entropy per patch."""
    model.eval()
    entropies = []
    with torch.no_grad():
        for index in range(0, len(pool_patches), Config.UNET_BATCH_SIZE):
            batch = torch.from_numpy(pool_patches[index:index + Config.UNET_BATCH_SIZE]).to(device)
            probabilities = F.softmax(model(batch), dim=1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            entropies.extend(entropy.mean(dim=(1, 2)).cpu().numpy())
    return np.asarray(entropies)


def manual_expert_labeling(patch_rgb, ref_mask, patch_id, uncertainty_score):
    """Display an informative patch and request its class from an expert."""
    plt.figure(figsize=(6, 3))
    rgb = np.stack([patch_rgb[2], patch_rgb[1], patch_rgb[0]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title(f"Patch #{patch_id} (Uncertainty: {uncertainty_score:.3f})")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(ref_mask, cmap="tab10", vmin=1, vmax=6)
    plt.title("MapBiomas Ground Truth")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("Target Classes:")
    for class_id, class_info in Config.CLASSES.items():
        print(f" [{class_id}] {class_info['name']}", end="  ")
    print()
    while True:
        try:
            value = input(
                f">> [EXPERT ORACLE] Assign class ID for Patch #{patch_id} "
                "(or press Enter to accept ground truth): "
            )
            if not value.strip():
                chosen_class = int(np.bincount(ref_mask.flatten(), minlength=7)[1:7].argmax() + 1)
                print(
                    f"Defaulting to dominant reference class: {chosen_class} "
                    f"({Config.CLASSES[chosen_class]['name']})"
                )
                return np.full_like(ref_mask, chosen_class)
            chosen_class = int(value)
            if chosen_class in Config.CLASSES:
                return np.full_like(ref_mask, chosen_class)
            print("Invalid input! Please enter a number between 1 and 6.")
        except (TypeError, ValueError):
            print("Invalid input! Please enter a number between 1 and 6.")


def plot_thematic_maps(area_name, season_name, s2_matrix, true_mask, pred_mask, run_tag=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rgb = np.stack([s2_matrix[2], s2_matrix[1], s2_matrix[0]], axis=-1)
    axes[0].imshow((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10))
    axes[0].set_title(f"Sentinel-2 Real RGB ({area_name} - {season_name} Season)")
    axes[0].axis("off")
    true_colors = np.zeros((*true_mask.shape, 3))
    pred_colors = np.zeros((*true_mask.shape, 3))
    for class_id, info in Config.CLASSES.items():
        color = np.array(
            [int(info["color"].lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        )
        true_colors[true_mask == class_id] = color
        pred_colors[pred_mask == class_id] = color
    axes[1].imshow(true_colors)
    axes[1].set_title("MapBiomas Reference Classes")
    axes[1].axis("off")
    axes[2].imshow(pred_colors)
    axes[2].set_title(f"U-Net Classification [{run_tag}]")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(Config.FIGURES_DIR, f"thematic_{area_name}_{season_name}_{run_tag}.png"),
        dpi=200,
    )
    plt.show()


def run_active_learning_cycle(x_pool, y_pool, model, device, area_name):
    """Train with seed samples and iteratively query uncertain patches."""
    if len(x_pool) == 0:
        return model, []
    seed_size = min(Config.SEED_SIZE, len(x_pool))
    train_indices = list(np.random.choice(len(x_pool), seed_size, replace=False))
    pool_indices = list(np.setdiff1d(np.arange(len(x_pool)), train_indices))
    working_labels = np.copy(y_pool)
    print(f"\n--- Initializing Model Training with {seed_size} Seed Samples [{area_name}] ---")

    for cycle in range(Config.ACTIVE_LEARNING_CYCLES):
        loader = DataLoader(
            SegmentationDataset(x_pool[train_indices], working_labels[train_indices]),
            batch_size=Config.UNET_BATCH_SIZE,
            shuffle=True,
        )
        train_unet_model(model, loader, epochs=Config.UNET_EPOCHS_PER_CYCLE, device=device)
        if not pool_indices:
            break
        print(f"\n[Active Learning Round {cycle + 1}/{Config.ACTIVE_LEARNING_CYCLES}] Computing U-Net entropy...")
        uncertainties = evaluate_unet_uncertainty(model, x_pool[pool_indices], device)
        query_count = min(Config.SAMPLES_PER_CYCLE, len(pool_indices))
        queried_positions = np.argsort(uncertainties)[-query_count:]
        queried_ids = [pool_indices[position] for position in queried_positions]
        print(f"Model queried {query_count} most uncertain patches for human expert annotation.")
        for query_id in queried_ids:
            score = float(uncertainties[pool_indices.index(query_id)])
            if Config.MANUAL_LABELING:
                working_labels[query_id] = manual_expert_labeling(
                    x_pool[query_id], y_pool[query_id], query_id, score
                )
            train_indices.append(query_id)
            pool_indices.remove(query_id)
    return model, train_indices


def evaluate_model_on_target(model, patches, masks, device):
    """Return overall accuracy, macro-F1, patch predictions, and per-patch F1."""
    model.eval()
    patch_predictions, patch_f1s = [], []
    with torch.no_grad():
        for index, patch in enumerate(patches):
            inputs = torch.from_numpy(patch[None]).to(device)
            prediction = torch.argmax(model(inputs), dim=1).cpu().numpy()[0] + 1
            patch_predictions.append(prediction)
            patch_f1s.append(
                f1_score(masks[index].ravel(), prediction.ravel(), average="macro", zero_division=0)
            )
    if not patch_predictions:
        return 0.0, 0.0, np.asarray(patch_predictions), np.asarray(patch_f1s)
    y_true = masks.flatten()
    y_pred = np.asarray(patch_predictions).flatten()
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
        np.asarray(patch_predictions),
        np.asarray(patch_f1s),
    )


def run_bidirectional_experiment():
    """Run both seasonal, cross-region train/test directions."""
    Config.create_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on compute device: {device}")
    data = {}
    for area in Config.study_areas():
        name = area["name"]
        data[name] = {}
        for season in ("Dry", "Wet"):
            x, y, s2_matrix, ref_matrix, coordinates = process_and_patch_seasonal(area, season)
            data[name][season] = {
                "x": x,
                "y": y,
                "s2": s2_matrix,
                "ref": ref_matrix,
                "coords": coordinates,
            }

    all_results, geostat_payload = [], {}
    for season in ("Dry", "Wet"):
        print("\n" + "#" * 75)
        print(f" DATASET SHIFT EXPERIMENT: {season.upper()} SEASON")
        print("#" * 75)
        crist, quer = data["Cristalina"][season], data["Querência"][season]
        for source_name, target_name, source, target in (
            ("Cristalina", "Querência", crist, quer),
            ("Querência", "Cristalina", quer, crist),
        ):
            print(f"\n>>> DIRECTION: {source_name} -> {target_name} [{season}]")
            split = int(len(source["x"]) * 0.7)
            x_train, y_train = source["x"][:split], source["y"][:split]
            x_test, y_test = source["x"][split:], source["y"][split:]
            model = UNet2D(len(Config.BANDS), len(Config.CLASSES)).to(device)
            model, _ = run_active_learning_cycle(
                x_train, y_train, model, device, f"{source_name}_{season}"
            )
            oa_intra, f1_intra, _, _ = evaluate_model_on_target(model, x_test, y_test, device)
            oa_inter, f1_inter, _, patch_f1s = evaluate_model_on_target(
                model, target["x"], target["y"], device
            )
            all_results.extend([
                {
                    "Source": source_name,
                    "Target": f"{source_name} (Intra)",
                    "Season": season,
                    "OA": oa_intra,
                    "Macro-F1": f1_intra,
                },
                {
                    "Source": source_name,
                    "Target": f"{target_name} (Inter)",
                    "Season": season,
                    "OA": oa_inter,
                    "Macro-F1": f1_inter,
                },
            ])
            source_key = "Cris" if source_name == "Cristalina" else "Quer"
            target_key = "Quer" if target_name == "Querência" else "Cris"
            geostat_payload[f"{source_key}->{target_key}_{season}"] = {
                "coords": target["coords"],
                "f1_values": patch_f1s,
            }

    metrics = pd.DataFrame(all_results)
    print("\n" + "=" * 25 + " BIDIRECTIONAL GENERALIZATION SUMMARY " + "=" * 25)
    print(metrics.to_string(index=False))
    if not metrics.empty:
        plt.figure(figsize=(11, 5))
        sns.barplot(data=metrics, x="Source", y="Macro-F1", hue="Target", palette="viridis")
        plt.ylim(0.4, 1.0)
        plt.title("Bidirectional Dataset Shift Impact: Cristalina $\\leftrightarrow$ Querência")
        plt.ylabel("Mean Macro F1-Score")
        plt.xlabel("Original Model Training Dataset Domain")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(Config.FIGURES_DIR, "bidirectional_shift_performance.png"), dpi=300)
        plt.show()
    return metrics, geostat_payload


if __name__ == "__main__":
    initialize_earth_engine()
    metrics, geostat_data = run_bidirectional_experiment()
