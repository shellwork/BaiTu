# ptc_utils.py
from typing import Any, Dict, List, Tuple, Union
import os

from PIL import Image as PILImage
import matplotlib.pyplot as plt


def build_source_wells_by_slot(
    source_slots: List[str],
    source_wells: List[List[str]],
) -> Dict[str, List[str]]:
    """
    Build mapping: source slot -> list of source wells.
    source_slots must be a list of slot names.
    source_wells must be a list of lists of well names.
    """
    if source_slots is None or source_wells is None:
        raise ValueError("source_slots and source_wells must not be None.")
    if not isinstance(source_slots, list) or not isinstance(source_wells, list):
        raise ValueError("source_slots must be a list and source_wells must be a list of lists.")
    if len(source_slots) == 0:
        raise ValueError("source_slots must not be empty.")
    if len(source_slots) != len(source_wells):
        raise ValueError("source_slots and source_wells must have the same length.")

    mapping: Dict[str, List[str]] = {}
    for slot, wells in zip(source_slots, source_wells):
        if not isinstance(wells, list):
            raise ValueError("Each element of source_wells must be a list of well names.")
        mapping[slot] = [str(w) for w in wells]
    return mapping


def build_dispense_wells_by_slot(
    dispense_slots: List[str],
    dispense_wells: Union[str, List[Union[str, List[str]]]],
) -> Dict[str, List[str]]:
    """
    Normalize dispense definition into a mapping: dispense slot -> list of wells.

    Supported formats:
        dispense_slots = ["10"] or ["10", "11"]

        dispense_wells = "A1"
            -> {slot: ["A1"] for slot in dispense_slots}

        dispense_wells = ["A1", "A2"]   (only when len(dispense_slots) == 1)
            -> {dispense_slots[0]: ["A1", "A2"]}

        dispense_wells = [["C1", "C2"], ["D1"]]
            -> {dispense_slots[0]: ["C1", "C2"], dispense_slots[1]: ["D1"]}
    """
    if dispense_slots is None or not isinstance(dispense_slots, list):
        raise ValueError("dispense_slots must be a non-empty list of slot names.")
    if len(dispense_slots) == 0:
        raise ValueError("dispense_slots must not be empty.")

    if dispense_wells is None:
        raise ValueError(
            "dispense_wells must be provided. "
            "Use a single well like 'A1', a flat list like ['A1', 'A2'] for a single plate, "
            "or a list-of-list for multiple plates."
        )

    # Case 1: single string -> same well on all plates
    if isinstance(dispense_wells, str):
        return {slot: [dispense_wells] for slot in dispense_slots}

    # From here on, we expect a list-like
    if not isinstance(dispense_wells, list):
        raise ValueError(
            f"Unsupported dispense_wells type: {type(dispense_wells)}. "
            "Expected str, ['A1', ...], or list-of-list like [['C1','C2'], ...]."
        )
    if len(dispense_wells) == 0:
        raise ValueError(
            "dispense_wells must not be an empty list. "
            "Pass a single well like ['A1'] or a list-of-list."
        )

    first = dispense_wells[0]

    # Case 2: flat list of strings
    if isinstance(first, str):
        if len(dispense_slots) == 1:
            slot = dispense_slots[0]
            wells = [str(w) for w in dispense_wells]
            return {slot: wells}
        raise ValueError(
            "Flat list dispense_wells (e.g. ['A1','A2']) is only allowed "
            "when there is exactly ONE dispense plate. "
            "For multiple plates, use a list-of-list like "
            "[['C1','C2'], ['D1']] that matches dispense_slots."
        )

    # Case 3: list-of-list (or list-of-tuple)
    if isinstance(first, (list, tuple)):
        if len(dispense_wells) != len(dispense_slots):
            raise ValueError(
                "When using list-of-list for dispense_wells, the outer length must "
                f"match number of dispense_slots. Got {len(dispense_wells)} lists "
                f"for {len(dispense_slots)} slots."
            )
        mapping: Dict[str, List[str]] = {}
        for slot, wells in zip(dispense_slots, dispense_wells):
            mapping[slot] = [str(w) for w in wells]
        return mapping

    raise ValueError(
        "Unsupported dispense_wells format. "
        "Expected str, ['A1', ...], or list-of-list of strings like "
        "[['C1','C2'], ['D1'], ...]."
    )


def load_labware_from_config(ot_module: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load pipette, tiprack, source labware, imaging labware, and dispense labware
    from a simple config dict.
    """
    result: Dict[str, Any] = {}

    # Pipette (channel/index 0)
    pipette_conf = config.get("pipette")
    if pipette_conf is not None:
        pipette_name = pipette_conf["name"]
        pipette_id = ot_module.load_equipment(0, pipette_name)
        print(f"[PTC] Pipette loaded: {pipette_id} (name={pipette_name})")
        result["pipette_id"] = pipette_id

    # Tiprack (channel/index 1, with slot)
    tiprack_conf = config.get("tiprack")
    if tiprack_conf is not None:
        tiprack_name = tiprack_conf["name"]
        tiprack_slot = tiprack_conf["slot"]
        tiprack_id = ot_module.load_equipment(1, tiprack_name, slot_name=tiprack_slot)
        print(f"[PTC] Tiprack loaded: {tiprack_id} (slot {tiprack_slot})")
        result["tiprack_id"] = tiprack_id

    # Sources (multiple slots -> one labware type)
    sources_conf = config.get("sources")
    if sources_conf is not None:
        source_name = sources_conf["name"]
        source_slots: List[str] = sources_conf["slots"]
        sources_by_slot: Dict[str, str] = {}
        for source_slot in source_slots:
            labware_id = ot_module.load_equipment(
                1,
                source_name,
                slot_name=source_slot,
            )
            print(f"[PTC] Source labware loaded: {labware_id} (slot {source_slot})")
            sources_by_slot[source_slot] = labware_id
        result["sources"] = sources_by_slot

    # Imaging labware
    imaging_conf = config.get("imaging")
    if imaging_conf is not None:
        imaging_name = imaging_conf["name"]
        imaging_slot = imaging_conf["slot"]
        imaging_labware_id = ot_module.load_equipment(
            1,
            imaging_name,
            slot_name=imaging_slot,
        )
        print(f"[PTC] Imaging labware loaded: {imaging_labware_id} (slot {imaging_slot})")
        result["imaging_labware_id"] = imaging_labware_id

    # Dispense labware (single or multiple plates)
    dispense_conf = config.get("dispense")
    if dispense_conf is not None:
        dispense_name = dispense_conf["name"]
        if "slots" in dispense_conf:
            dispense_slots = dispense_conf["slots"]
        elif "slot" in dispense_conf:
            dispense_slots = [dispense_conf["slot"]]
        else:
            raise ValueError(
                "dispense config must contain either 'slot' (single) or 'slots' (list of slots)."
            )

        dispenses_by_slot: Dict[str, str] = {}
        for slot in dispense_slots:
            dispense_labware_id = ot_module.load_equipment(
                1,
                dispense_name,
                slot_name=slot,
            )
            print(f"[PTC] Dispense labware loaded: {dispense_labware_id} (slot {slot})")
            dispenses_by_slot[slot] = dispense_labware_id
        result["dispenses"] = dispenses_by_slot

        if len(dispense_slots) == 1:
            # Convenience field for single-plate cases
            result["dispense_labware_id"] = next(iter(dispenses_by_slot.values()))

    return result


def run_liquid_level_check(
    ot_module: Any,
    vision_module: Any,
    model: Any,
    run_id: str,
    imaging_labware_id: str,
    imaging_well: str,
    imaging_offset: Tuple[float, float, float],
    base_dir: str,
    source_slot: str,
    source_well: str,
    conf_threshold: float,
    expected_tips: int,
    expected_vol: float,
    show_images: bool = True,
):
    """
    Move to imaging position, capture an image, run the vision model, and
    optionally display images. Returns (image_path, prediction_path, result_dict).
    """
    print(f"[PTC] Moving to imaging position: {imaging_well}...")
    ot_module.move(
        imaging_labware_id,
        wellname=imaging_well,
        offset=imaging_offset,
    )
    print("[PTC] Reached imaging position.")

    print("[PTC] Capturing image...")
    image_path = vision_module.capture_image_with_run_id(
        run_id=run_id,
        step_name=f"liquid_check_{source_slot}_{source_well}",
        base_dir=base_dir,
    )
    print(f"[PTC] Image saved: {image_path}")

    print("[PTC] Running vision model...")
    result = vision_module.check_liquid_level(
        model=model,
        image_path=image_path,
        conf_threshold=conf_threshold,
        expected_tips=expected_tips,
        expected_vol=expected_vol,
    )

    prediction_path = result.get("prediction_image_path")
    detected_levels = result.get("detected_levels", [])
    pass_status = result.get("channel_pass_status", [])
    expected_height_percent = result.get("expected_height_percent", None)

    print("=" * 60)
    print("  LIQUID LEVEL CHECK SUMMARY")
    print("=" * 60)
    print(f"  Source slot: {source_slot}, well: {source_well}")
    print(f"  Target Volume: {expected_vol} uL")
    if expected_height_percent is not None:
        print(f"  Expected Height (approx): {expected_height_percent:.2f}%")
    print(f"  Detected Levels: {detected_levels}")
    print(f"  Pass Status: {pass_status}")
    print(f"  Passed: {result.get('passed', False)}")
    print("=" * 60)

    if show_images:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            if image_path and os.path.exists(image_path):
                img = PILImage.open(image_path)
                axes[0].imshow(img)
                axes[0].set_title(f"Original ({source_slot}:{source_well})")
                axes[0].axis("off")
            else:
                axes[0].text(0.5, 0.5, "Image not found", ha="center", va="center")
                axes[0].axis("off")

            if prediction_path and os.path.exists(prediction_path):
                pred_img = PILImage.open(prediction_path)
                axes[1].imshow(pred_img)
                axes[1].set_title("Prediction")
                axes[1].axis("off")
            else:
                axes[1].text(0.5, 0.5, "Prediction not available", ha="center", va="center")
                axes[1].axis("off")

            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"[PTC] WARNING: Could not display images: {exc}")

    return image_path, prediction_path, result


def auto_recover_tips(
    ot_module: Any,
    tiprack_id: str,
    pick_well: str,
    move_height: float = 20.0,
    drop_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bool:
    """
    Return tips to the original pick well and home the robot.
    """
    try:
        print(f"[PTC] Moving back above original pick well {pick_well}...")
        ot_module.move(tiprack_id, wellname=pick_well, offset=(0.0, 0.0, move_height))
        print(f"[PTC] Reached position {pick_well}")

        print(f"[PTC] Dropping tips back to original well {pick_well}...")
        ot_module.drop_tips(
            tiprack_id=tiprack_id,
            wellname=pick_well,
            offset=drop_offset,
        )
        print(f"[PTC] Tips dropped to {pick_well}")

        print("[PTC] Homing robot after tip recovery...")
        ot_module.home()
        print("[PTC] Home position reached.")
        return True

    except Exception as exc:
        print(f"[PTC WARNING] Tip auto-recovery failed: {exc}")
        return False


def auto_recover_liquid_and_tips(
    ot_module: Any,
    source_labware_id: str,
    source_well: str,
    tiprack_id: str,
    pick_well: str,
    aspirate_vol: float,
    source_dispense_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    blow_out_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    pick_move_height: float = 20.0,
) -> bool:
    """
    Recover from a failed liquid-level check:
      1) Return liquid to the source well (dispense + blow out).
      2) Return tips to the original pick well.
    """
    try:
        print(f"[PTC] Dispensing {aspirate_vol} uL back to source well {source_well}...")
        ot_module.dispense(
            volume=aspirate_vol,
            labware_id=source_labware_id,
            wellname=source_well,
            offset=source_dispense_offset,
        )

        print("[PTC] Performing blow out in source well...")
        ot_module.blow_out(
            labware_id=source_labware_id,
            wellname=source_well,
            offset=blow_out_offset,
        )
        print("[PTC] Liquid recovery step completed.")

        print("[PTC] Starting tip recovery after LLD failure...")
        tips_ok = auto_recover_tips(
            ot_module=ot_module,
            tiprack_id=tiprack_id,
            pick_well=pick_well,
            move_height=pick_move_height,
            drop_offset=(0.0, 0.0, 0.0),
        )
        return tips_ok

    except Exception as exc:
        print(f"[PTC WARNING] Liquid recovery failed: {exc}")
        return False
