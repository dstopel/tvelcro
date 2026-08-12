#!/bin/bash
set -euo pipefail

# ==============================================================================
# Create image groups from ILSVRC2012_val_by_name
#
# Usage:
#   bash create_groups.sh \
#     --src /path/to/ILSVRC2012_val_by_name \
#     --dst /path/to/image_for_exp \
#     --num_profile 300 \
#     --num_eval 300
# ==============================================================================

SRC=""
DST=""
NUM_PROFILE=300
NUM_EVAL=300

while [[ $# -gt 0 ]]; do
    case $1 in
        --src) SRC="$2"; shift 2 ;;
        --dst) DST="$2"; shift 2 ;;
        --num_profile) NUM_PROFILE="$2"; shift 2 ;;
        --num_eval) NUM_EVAL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; echo "Usage: bash create_groups.sh --src <dir> --dst <dir> [--num_profile N] [--num_eval N]"; exit 1 ;;
    esac
done

if [ -z "${SRC}" ] || [ -z "${DST}" ]; then
    echo "Usage: bash create_groups.sh --src <ILSVRC2012_val_by_name> --dst <image_for_exp> [--num_profile 300] [--num_eval 300]"
    exit 1
fi

if [ ! -d "${SRC}" ]; then
    echo "ERROR: Source dir not found: ${SRC}"
    exit 1
fi

SRC="$(cd "${SRC}" && pwd)"

echo "Source:      ${SRC}"
echo "Target:      ${DST}"
echo "Num profile: ${NUM_PROFILE}"
echo "Num eval:    ${NUM_EVAL}"

# ==============================================================================
# Group definitions
# ==============================================================================

SNAKES_LIST=(
    "thunder_snake_worm_snake_Carphophis_amoenus"
    "ringneck_snake_ring-necked_snake_ring_snake"
    "hognose_snake_puff_adder_sand_viper"
    "green_snake_grass_snake"
    "king_snake_kingsnake"
    "garter_snake_grass_snake"
    "water_snake"
    "vine_snake"
    "night_snake_Hypsiglena_torquata"
    "boa_constrictor_Constrictor_constrictor"
    "rock_python_rock_snake_Python_sebae"
    "Indian_cobra_Naja_naja"
    "green_mamba"
    "sea_snake"
    "horned_viper_cerastes_sand_viper_horned_asp_Cerastes_cornutus"
    "diamondback_diamondback_rattlesnake_Crotalus_adamanteus"
    "sidewinder_horned_rattlesnake_Crotalus_cerastes"
)

FISH_LIST=(
    "tench_Tinca_tinca"
    "goldfish_Carassius_auratus"
    "great_white_shark_white_shark_man-eater_man-eating_shark_Carcharodon_carcharias"
    "tiger_shark_Galeocerdo_cuvieri"
    "hammerhead_hammerhead_shark"
    "electric_ray_crampfish_numbfish_torpedo"
    "stingray"
    "barracouta_snoek"
    "eel"
    "coho_cohoe_coho_salmon_blue_jack_silver_salmon_Oncorhynchus_kisutch"
    "rock_beauty_Holocanthus_tricolor"
    "anemone_fish"
    "sturgeon"
    "gar_garfish_garpike_billfish_Lepisosteus_osseus"
    "lionfish"
    "puffer_pufferfish_blowfish_globefish"
)

BOATS_LIST=(
    "canoe"
    "catamaran"
    "schooner"
    "lifeboat"
    "gondola"
    "yawl"
    "fireboat"
    "pirate_pirate_ship"
    "aircraft_carrier_carrier_flattop_attack_aircraft_carrier"
    "container_ship_containership_container_vessel"
    "liner_ocean_liner"
    "speedboat"
    "trimaran"
    "submarine_pigboat_sub_U-boat"
    "boathouse"
    "paddlewheel_paddle_wheel"
)

BIRDS_LIST=(
    "cock"
    "hen"
    "ostrich_Struthio_camelus"
    "brambling_Fringilla_montifringilla"
    "goldfinch_Carduelis_carduelis"
    "house_finch_linnet_Carpodacus_mexicanus"
    "junco_snowbird"
    "indigo_bunting_indigo_finch_indigo_bird_Passerina_cyanea"
    "robin_American_robin_Turdus_migratorius"
    "bulbul"
    "jay"
    "magpie"
    "chickadee"
    "water_ouzel_dipper"
    "kite"
    "bald_eagle_American_eagle_Haliaeetus_leucocephalus"
    "vulture"
    "great_grey_owl_great_gray_owl_Strix_nebulosa"
    "black_grouse"
    "ptarmigan"
    "ruffed_grouse_partridge_Bonasa_umbellus"
    "prairie_chicken_prairie_grouse_prairie_fowl"
    "peacock"
    "quail"
    "partridge"
    "African_grey_African_gray_Psittacus_erithacus"
    "macaw"
    "sulphur-crested_cockatoo_Kakatoe_galerita_Cacatua_galerita"
    "lorikeet"
    "coucal"
    "bee_eater"
    "hornbill"
    "hummingbird"
    "jacamar"
    "toucan"
    "drake"
    "red-breasted_merganser_Mergus_serrator"
    "goose"
    "black_swan_Cygnus_atratus"
    "white_stork_Ciconia_ciconia"
    "black_stork_Ciconia_nigra"
    "spoonbill"
    "flamingo"
    "little_blue_heron_Egretta_caerulea"
    "American_egret_great_white_heron_Egretta_albus"
    "bittern"
    "crane"
    "limpkin_Aramus_pictus"
    "European_gallinule_Porphyrio_porphyrio"
    "American_coot_marsh_hen_mud_hen_water_hen_Fulica_americana"
    "bustard"
    "ruddy_turnstone_Arenaria_interpres"
    "red-backed_sandpiper_dunlin_Erolia_alpina"
    "redshank_Tringa_totanus"
    "dowitcher"
    "oystercatcher_oyster_catcher"
    "pelican"
    "king_penguin_Aptenodytes_patagonica"
    "albatross_mollymawk"
)

# ==============================================================================
# Helper: create group with profile/eval split, capped at NUM_PROFILE/NUM_EVAL
# ==============================================================================
create_group() {
    local group_name=$1
    shift
    local classes=("$@")

    echo ""
    echo "=== Creating group: ${group_name} (${#classes[@]} classes) ==="

    local full_dir="${DST}/${group_name}/val"
    local prof_dir="${DST}/${group_name}_profile/val"
    local eval_dir="${DST}/${group_name}_eval/val"

    mkdir -p "${full_dir}" "${prof_dir}" "${eval_dir}"

    local found=0
    local missing=0
    local prof_count=0
    local eval_count=0

    for cls in "${classes[@]}"; do
        local src_cls="${SRC}/${cls}"
        if [ ! -d "${src_cls}" ]; then
            echo "  [WARN] Not found: ${cls}"
            missing=$((missing + 1))
            continue
        fi
        found=$((found + 1))

        # Full group: symlink entire class dir
        ln -sfn "${src_cls}" "${full_dir}/${cls}"

        # Split images: odd-numbered -> profile, even-numbered -> eval
        # This interleaves so both splits get diverse images from each class
        local img_idx=0
        mkdir -p "${prof_dir}/${cls}" "${eval_dir}/${cls}"

        for img in "${src_cls}"/*; do
            [ -f "${img}" ] || continue
            img_idx=$((img_idx + 1))

            if [ $((img_idx % 2)) -eq 1 ] && [ ${prof_count} -lt ${NUM_PROFILE} ]; then
                ln -sfn "${img}" "${prof_dir}/${cls}/$(basename "${img}")"
                prof_count=$((prof_count + 1))
            elif [ ${eval_count} -lt ${NUM_EVAL} ]; then
                ln -sfn "${img}" "${eval_dir}/${cls}/$(basename "${img}")"
                eval_count=$((eval_count + 1))
            elif [ ${prof_count} -lt ${NUM_PROFILE} ]; then
                ln -sfn "${img}" "${prof_dir}/${cls}/$(basename "${img}")"
                prof_count=$((prof_count + 1))
            fi

            if [ ${prof_count} -ge ${NUM_PROFILE} ] && [ ${eval_count} -ge ${NUM_EVAL} ]; then
                break 2
            fi
        done
    done

    # Remove empty class dirs
    find "${prof_dir}" -type d -empty -delete 2>/dev/null || true
    find "${eval_dir}" -type d -empty -delete 2>/dev/null || true

    echo "  Found: ${found} classes, Missing: ${missing}"
    echo "  Profile: ${prof_count} images, Eval: ${eval_count} images"
}

# ==============================================================================
# Create all groups
# ==============================================================================

create_group "snakes" "${SNAKES_LIST[@]}"
create_group "fish"   "${FISH_LIST[@]}"
create_group "boats"  "${BOATS_LIST[@]}"
create_group "birds"  "${BIRDS_LIST[@]}"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "========================================"
echo "DONE. New groups in ${DST}/:"
echo "========================================"
for g in snakes fish boats birds; do
    prof=$(find "${DST}/${g}_profile/val" -type l 2>/dev/null | wc -l)
    eval_c=$(find "${DST}/${g}_eval/val" -type l 2>/dev/null | wc -l)
    echo "  ${g}: ${prof} profile + ${eval_c} eval"
done
echo "========================================"
