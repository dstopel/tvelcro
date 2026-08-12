#!/usr/bin/env python3
"""
create_groups.py — build domain group dirs from an ImageNet val-by-class tree.

Replaces _legacy_giant/create_groups.sh and fixes its bugs:
  - Quotas are PER-CLASS: q = ceil(N / num_classes) per class, so ALL classes of
    a domain are covered (the legacy per-group counter with `break 2` sampled
    only the first ~12 of 59 bird classes).
  - Deterministic seeded selection (per-class shuffle keyed by seed/domain/class).
  - Profile/eval splits are disjoint by construction (shuffle once, slice twice).

Output layout (what the experiment scripts expect):
  DST/<domain>_profile/val/<class>/<symlink>
  DST/<domain>_eval/val/<class>/<symlink>
  DST/groups_manifest.json   (documents exactly which classes each domain includes)

Domains:
  snakes/fish/boats/birds : class-folder lists kept from the legacy script
  dogs   : ImageNet dog-breed classes (indices 151-268)
  cats   : felid classes (indices 281-293)
  cars   : motor-vehicle classes (exact-synonym match in synset_words.txt)
  general: uniform sample across all 1000 classes
The synset-derived domains require --synset_words.
"""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path

# Legacy class-folder lists (ILSVRC2012_val_by_name naming: synonyms joined by _)
SNAKES = [
    "thunder_snake_worm_snake_Carphophis_amoenus",
    "ringneck_snake_ring-necked_snake_ring_snake",
    "hognose_snake_puff_adder_sand_viper",
    "green_snake_grass_snake",
    "king_snake_kingsnake",
    "garter_snake_grass_snake",
    "water_snake",
    "vine_snake",
    "night_snake_Hypsiglena_torquata",
    "boa_constrictor_Constrictor_constrictor",
    "rock_python_rock_snake_Python_sebae",
    "Indian_cobra_Naja_naja",
    "green_mamba",
    "sea_snake",
    "horned_viper_cerastes_sand_viper_horned_asp_Cerastes_cornutus",
    "diamondback_diamondback_rattlesnake_Crotalus_adamanteus",
    "sidewinder_horned_rattlesnake_Crotalus_cerastes",
]
FISH = [
    "tench_Tinca_tinca",
    "goldfish_Carassius_auratus",
    "great_white_shark_white_shark_man-eater_man-eating_shark_Carcharodon_carcharias",
    "tiger_shark_Galeocerdo_cuvieri",
    "hammerhead_hammerhead_shark",
    "electric_ray_crampfish_numbfish_torpedo",
    "stingray",
    "barracouta_snoek",
    "eel",
    "coho_cohoe_coho_salmon_blue_jack_silver_salmon_Oncorhynchus_kisutch",
    "rock_beauty_Holocanthus_tricolor",
    "anemone_fish",
    "sturgeon",
    "gar_garfish_garpike_billfish_Lepisosteus_osseus",
    "lionfish",
    "puffer_pufferfish_blowfish_globefish",
]
BOATS = [
    "canoe",
    "catamaran",
    "schooner",
    "lifeboat",
    "gondola",
    "yawl",
    "fireboat",
    "pirate_pirate_ship",
    "aircraft_carrier_carrier_flattop_attack_aircraft_carrier",
    "container_ship_containership_container_vessel",
    "liner_ocean_liner",
    "speedboat",
    "trimaran",
    "submarine_pigboat_sub_U-boat",
    "boathouse",
    "paddlewheel_paddle_wheel",
]
BIRDS = [
    "cock",
    "hen",
    "ostrich_Struthio_camelus",
    "brambling_Fringilla_montifringilla",
    "goldfinch_Carduelis_carduelis",
    "house_finch_linnet_Carpodacus_mexicanus",
    "junco_snowbird",
    "indigo_bunting_indigo_finch_indigo_bird_Passerina_cyanea",
    "robin_American_robin_Turdus_migratorius",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water_ouzel_dipper",
    "kite",
    "bald_eagle_American_eagle_Haliaeetus_leucocephalus",
    "vulture",
    "great_grey_owl_great_gray_owl_Strix_nebulosa",
    "black_grouse",
    "ptarmigan",
    "ruffed_grouse_partridge_Bonasa_umbellus",
    "prairie_chicken_prairie_grouse_prairie_fowl",
    "peacock",
    "quail",
    "partridge",
    "African_grey_African_gray_Psittacus_erithacus",
    "macaw",
    "sulphur-crested_cockatoo_Kakatoe_galerita_Cacatua_galerita",
    "lorikeet",
    "coucal",
    "bee_eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted_merganser_Mergus_serrator",
    "goose",
    "black_swan_Cygnus_atratus",
    "white_stork_Ciconia_ciconia",
    "black_stork_Ciconia_nigra",
    "spoonbill",
    "flamingo",
    "little_blue_heron_Egretta_caerulea",
    "American_egret_great_white_heron_Egretta_albus",
    "bittern",
    "crane",
    "limpkin_Aramus_pictus",
    "European_gallinule_Porphyrio_porphyrio",
    "American_coot_marsh_hen_mud_hen_water_hen_Fulica_americana",
    "bustard",
    "ruddy_turnstone_Arenaria_interpres",
    "red-backed_sandpiper_dunlin_Erolia_alpina",
    "redshank_Tringa_totanus",
    "dowitcher",
    "oystercatcher_oyster_catcher",
    "pelican",
    "king_penguin_Aptenodytes_patagonica",
    "albatross_mollymawk",
]

# ImageNet index ranges (canonical): dog breeds 151-268, felids 281-293
DOG_INDICES = range(151, 269)
CAT_INDICES = range(281, 294)

# Motor-vehicle classes: matched by EXACT synonym in synset_words.txt (avoids
# substring hits like "cardigan"). Curated list; the emitted manifest documents
# exactly which classes matched.
CAR_SYNONYMS = {
    "ambulance", "beach wagon", "cab", "convertible", "fire engine",
    "garbage truck", "go-kart", "golfcart", "jeep", "limousine", "minibus",
    "minivan", "model t", "moped", "motor scooter", "moving van", "pickup",
    "police van", "racer", "recreational vehicle", "school bus", "snowplow",
    "sports car", "tow truck", "trailer truck", "trolleybus",
}

ALL_DOMAINS = ["snakes", "fish", "boats", "birds", "dogs", "cats", "cars", "general"]
SYNSET_DOMAINS = {"dogs", "cats", "cars", "general"}


def parse_synsets(synset_words_path):
    """Returns list of dicts: {idx, wnid, synonyms, folder} per ImageNet class."""
    entries = []
    with open(synset_words_path) as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            synonyms = [s.strip() for s in parts[1].split(",")]
            folder = "_".join(s.replace(" ", "_") for s in synonyms)
            entries.append({"idx": idx, "wnid": parts[0],
                            "synonyms": synonyms, "folder": folder})
    return entries


def resolve_class_dir(src: Path, entry_or_name):
    """Find the actual class dir under src (joined-name, first-synonym, or wnid)."""
    if isinstance(entry_or_name, str):
        candidates = [entry_or_name]
    else:
        candidates = [entry_or_name["folder"],
                      entry_or_name["synonyms"][0].replace(" ", "_"),
                      entry_or_name["wnid"]]
    for c in candidates:
        if (src / c).is_dir():
            return c
    return None


def domain_classes(domain, src: Path, synsets):
    """Returns (resolved class dir names, class metadata list, missing list)."""
    if domain in ("snakes", "fish", "boats", "birds"):
        wanted = {"snakes": SNAKES, "fish": FISH, "boats": BOATS, "birds": BIRDS}[domain]
        by_folder = {e["folder"]: e for e in synsets} if synsets else {}
        resolved, meta, missing = [], [], []
        for name in wanted:
            r = resolve_class_dir(src, name)
            if r is None:
                missing.append(name)
                continue
            resolved.append(r)
            e = by_folder.get(name)
            meta.append({"class_dir": r, "imagenet_idx": e["idx"] if e else None,
                         "name": e["synonyms"][0] if e else name})
        return resolved, meta, missing

    if synsets is None:
        raise ValueError(f"domain '{domain}' requires --synset_words")

    if domain == "dogs":
        wanted = [e for e in synsets if e["idx"] in DOG_INDICES]
    elif domain == "cats":
        wanted = [e for e in synsets if e["idx"] in CAT_INDICES]
    elif domain == "cars":
        wanted = [e for e in synsets
                  if any(s.lower() in CAR_SYNONYMS for s in e["synonyms"])]
    else:  # general
        wanted = list(synsets)

    resolved, meta, missing = [], [], []
    for e in wanted:
        r = resolve_class_dir(src, e)
        if r is None:
            missing.append(e["folder"])
            continue
        resolved.append(r)
        meta.append({"class_dir": r, "imagenet_idx": e["idx"],
                     "name": e["synonyms"][0]})
    return resolved, meta, missing


def round_robin_cap(picks_per_class, class_order, cap):
    """Interleave one pick per class per round, truncate at cap.
    Guarantees every class is represented as long as cap >= num_classes."""
    out = []
    max_len = max((len(v) for v in picks_per_class.values()), default=0)
    for k in range(max_len):
        for cls in class_order:
            picks = picks_per_class[cls]
            if k < len(picks):
                out.append((cls, picks[k]))
    return out[:cap]


def build_domain(domain, src: Path, dst: Path, classes, num_profile, num_eval, seed):
    n_classes = len(classes)
    q_prof = math.ceil(num_profile / n_classes)
    q_eval = math.ceil(num_eval / n_classes)

    prof_picks, eval_picks = {}, {}
    for cls in classes:
        files = sorted(p.name for p in (src / cls).iterdir() if p.is_file())
        rng = random.Random(f"{seed}:{domain}:{cls}")
        rng.shuffle(files)
        # Disjoint by construction: one shuffle, two slices
        prof_picks[cls] = files[:q_prof]
        eval_picks[cls] = files[q_prof:q_prof + q_eval]

    class_order = sorted(classes)
    splits = {
        f"{domain}_profile": round_robin_cap(prof_picks, class_order, num_profile),
        f"{domain}_eval": round_robin_cap(eval_picks, class_order, num_eval),
    }

    counts = {}
    for split_name, picks in splits.items():
        split_root = dst / split_name
        if split_root.exists():
            shutil.rmtree(split_root)  # deterministic rebuild, no stale links
        for cls, fname in picks:
            cls_dir = split_root / "val" / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            os.symlink((src / cls / fname).resolve(), cls_dir / fname)
        counts[split_name] = len(picks)

    return {"per_class_quota_profile": q_prof, "per_class_quota_eval": q_eval,
            "profile_count": counts[f"{domain}_profile"],
            "eval_count": counts[f"{domain}_eval"]}


def main():
    ap = argparse.ArgumentParser(description="Build T-VELCRO domain group dirs "
                                             "from an ImageNet val-by-class tree")
    ap.add_argument("--src", default=os.environ.get("IMAGENET_DIR"),
                    help="ImageNet val-by-class tree (default: $IMAGENET_DIR)")
    ap.add_argument("--dst", required=True, help="Output dir for group dirs")
    ap.add_argument("--synset_words", default=None,
                    help="synset_words.txt (required for dogs/cats/cars/general)")
    ap.add_argument("--num_profile", type=int, default=300)
    ap.add_argument("--num_eval", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domains", default=",".join(ALL_DOMAINS),
                    help=f"Comma-separated subset of: {','.join(ALL_DOMAINS)}")
    args = ap.parse_args()

    if not args.src:
        raise SystemExit("--src or $IMAGENET_DIR required")
    src = Path(args.src).resolve()
    if not src.is_dir():
        raise SystemExit(f"Source dir not found: {src}")
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = set(domains) - set(ALL_DOMAINS)
    if unknown:
        raise SystemExit(f"Unknown domains: {sorted(unknown)}")
    if set(domains) & SYNSET_DOMAINS and not args.synset_words:
        raise SystemExit("--synset_words required for dogs/cats/cars/general")

    synsets = parse_synsets(args.synset_words) if args.synset_words else None

    manifest = {
        "seed": args.seed,
        "num_profile": args.num_profile,
        "num_eval": args.num_eval,
        "src": str(src),
        "split_rule": "per-class seeded shuffle; profile = first ceil(N/C) picks, "
                      "eval = next ceil(N/C) picks (disjoint); round-robin "
                      "truncation to N covers all classes when N >= C",
        "domains": {},
    }

    for domain in domains:
        print(f"\n=== {domain} ===")
        classes, meta, missing = domain_classes(domain, src, synsets)
        if missing:
            print(f"  [WARN] {len(missing)} class dir(s) not found: {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
        if not classes:
            print("  [SKIP] no classes resolved")
            continue
        stats = build_domain(domain, src, dst, classes,
                             args.num_profile, args.num_eval, args.seed)
        print(f"  {len(classes)} classes, quota {stats['per_class_quota_profile']}"
              f"/class -> {stats['profile_count']} profile + "
              f"{stats['eval_count']} eval images")
        manifest["domains"][domain] = {
            "num_classes": len(classes),
            "missing_classes": missing,
            **stats,
            "classes": meta,
        }

    manifest_path = dst / "groups_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
