import requests
import json
from typing import Any, Dict, Optional

# OT-2 connection settings
HEADERS = {"Opentrons-Version": "3"}
ROBOT_IP = "169.254.200.128"   # Change this if your robot IP changes

# Global state for current run
commands_url: Optional[str] = None
pipette_id: Optional[str] = None
LABWARE_BY_SLOT: Dict[str, str] = {}


def reconnect_last_run(prefer_mount: str = "right"):
    """
    Reconnect to the most recent run on the OT-2 and recover:
      - commands_url  (for posting commands)
      - pipette_id    (for pipette-related helpers)
      - LABWARE_BY_SLOT (slotName -> labwareId mapping)

    Returns:
        dict with keys: run_id, pipette_id, labware_by_slot
    """
    global commands_url, pipette_id, LABWARE_BY_SLOT

    base = f"http://{ROBOT_IP}:31950/runs"
    print(f"[OT] GET {base}")
    r = requests.get(base, headers=HEADERS)
    r.raise_for_status()

    payload = r.json()
    runs = payload.get("data", [])
    if not runs:
        raise RuntimeError("No runs found on the robot.")

    # Use the last run as the most recent one
    last_run = runs[-1]
    run_id = last_run["id"]
    commands_url = f"{base}/{run_id}/commands"
    print(f"[OT] Reconnected to run: {run_id}")

    # ---------- Recover pipette_id ----------
    pipette_id = None
    pipettes = last_run.get("pipettes", [])
    print(f"[OT] Found {len(pipettes)} pipette entries in run.")

    chosen_pipette = None
    for p in pipettes:
        if p.get("mount") == prefer_mount:
            chosen_pipette = p
            break
    if chosen_pipette is None and pipettes:
        chosen_pipette = pipettes[0]

    if chosen_pipette is not None:
        pipette_id = chosen_pipette.get("id")
        print(f"[OT] Recovered pipette_id from run: {pipette_id}")
    else:
        print("[OT][WARN] No pipette info found in run. pipette_id will remain None.")

    # ---------- Recover labware mapping ----------
    LABWARE_BY_SLOT.clear()
    labware_list = last_run.get("labware", [])
    print(f"[OT] Found {len(labware_list)} labware entries in run.")

    for lw in labware_list:
        lw_id = lw.get("id")
        location = lw.get("location") or {}
        slot_name = location.get("slotName") or location.get("slot_name")

        if lw_id and slot_name:
            LABWARE_BY_SLOT[slot_name] = lw_id

    print(f"[OT] LABWARE_BY_SLOT restored: {LABWARE_BY_SLOT}")

    state = {
        "run_id": run_id,
        "pipette_id": pipette_id,
        "labware_by_slot": dict(LABWARE_BY_SLOT),
    }
    return state


def get_labware_id_by_slot(slot_name: str) -> str:
    """Return the labwareId for a given deck slot."""
    if slot_name in LABWARE_BY_SLOT:
        return LABWARE_BY_SLOT[slot_name]

    raise KeyError(
        f"No labwareId recorded for slot '{slot_name}'. "
        f"Known slots: {list(LABWARE_BY_SLOT.keys())}"
    )

def _post_command(command_dict, wait=True):
    """Internal helper to post a command to the current run."""
    global commands_url

    if not commands_url:
        raise RuntimeError("No active run. Call create_run() first.")

    params = {"waitUntilComplete": True} if wait else None

    r = requests.post(
        url=commands_url,
        headers=HEADERS,
        json=command_dict,
        params=params
    )

    if not r.ok:
        print("\n[OT][HTTP ERROR]", r.status_code)
        try:
            print("[OT][RESPONSE TEXT]:")
            print(r.text)
        except Exception:
            pass
        r.raise_for_status()

    data = r.json()["data"]

    # Check command execution status
    cmd_status = data.get("status", "unknown")
    if cmd_status == "failed":
        error_info = data.get("error", {})
        error_type = error_info.get("errorType", "unknown")
        error_detail = error_info.get("detail", "no detail")
        print(f"\n[OT][COMMAND FAILED] {error_type}: {error_detail}")
        raise RuntimeError(f"OT-2 command failed: {error_type} — {error_detail}")

    return data


def create_run():
    """Create a new run and store its commands_url globally."""
    global commands_url

    runs_url = f"http://{ROBOT_IP}:31950/runs"
    print(f"[OT] POST {runs_url}")

    r = requests.post(url=runs_url, headers=HEADERS)
    r.raise_for_status()

    data = r.json()["data"]
    run_id = data["id"]
    commands_url = f"{runs_url}/{run_id}/commands"

    print(f"[OT] Run created: {run_id}")
    return run_id, commands_url

def load_equipment(equipment_type, equipment_name, slot_name=None):
    """
    Load pipette or labware.
    equipment_type: 0 -> pipette, 1 -> labware
    """
    global pipette_id

    if equipment_type == 0:
        cmd = {
            "data": {
                "commandType": "loadPipette",
                "params": {
                    "pipetteName": equipment_name,
                    "mount": "right"
                },
                "intent": "setup"
            }
        }
        print(f"[OT] loadPipette: {equipment_name} (right)")
        data = _post_command(cmd, wait=True)
        pipette_id = data["result"]["pipetteId"]
        print(f"[OT] Pipette ID: {pipette_id}")
        return pipette_id

    elif equipment_type == 1:
        if slot_name is None:
            raise ValueError("slot_name is required for labware")

        cmd = {
            "data": {
                "commandType": "loadLabware",
                "params": {
                    "location": {"slotName": slot_name},
                    "loadName": equipment_name,
                    "namespace": "opentrons",
                    "version": 1
                },
                "intent": "setup"
            }
        }
        print(f"[OT] loadLabware: {equipment_name} in slot {slot_name}")
        data = _post_command(cmd, wait=True)
        labware_id = data["result"]["labwareId"]
        print(f"[OT] Labware ID: {labware_id}")

        LABWARE_BY_SLOT[slot_name] = labware_id
        return labware_id

    else:
        raise ValueError("equipment_type must be 0 (pipette) or 1 (labware)")

def pick_up(tiprack_id, wellname=None, offset=None):
    """Pick up a tip from the given tiprack well."""
    if pipette_id is None:
        raise RuntimeError("Pipette not loaded.")

    well = wellname or "A1"
    off = offset or (0, 0, 0)

    cmd = {
        "data": {
            "commandType": "pickUpTip",
            "params": {
                "labwareId": tiprack_id,
                "wellName": well,
                "wellLocation": {
                    "origin": "top",
                    "offset": {"x": off[0], "y": off[1], "z": off[2]}
                },
                "pipetteId": pipette_id
            },
            "intent": "setup"
        }
    }
    print(f"[OT] pickUpTip: labware={tiprack_id}, well={well}, offset={off}")
    _post_command(cmd, wait=True)

def move(labware_id, wellname=None, offset=None):
    """Move pipette to a well."""
    if pipette_id is None:
        raise RuntimeError("Pipette not loaded.")

    well = wellname or "A1"
    off = offset or (0, 0, 0)

    cmd = {
        "data": {
            "commandType": "moveToWell",
            "params": {
                "labwareId": labware_id,
                "wellName": well,
                "wellLocation": {
                    "origin": "top",
                    "offset": {"x": off[0], "y": off[1], "z": off[2]}
                },
                "pipetteId": pipette_id
            },
            "intent": "setup"
        }
    }
    print(f"[OT] moveToWell: labware={labware_id}, well={well}, offset={off}")
    _post_command(cmd, wait=True)

def drop_tips(tiprack_id=None, wellname=None, offset=None):
    """Drop tip either back into a tiprack or into fixedTrash."""
    if pipette_id is None:
        raise RuntimeError("Pipette not loaded.")

    well = wellname or "A1"
    off = offset or (0, 0, 0)
    labware_id = tiprack_id or "fixedTrash"

    cmd = {
        "data": {
            "commandType": "dropTip",
            "params": {
                "labwareId": labware_id,
                "wellName": well,
                "wellLocation": {
                    "origin": "top",
                    "offset": {"x": off[0], "y": off[1], "z": off[2]}
                },
                "pipetteId": pipette_id
            },
            "intent": "setup"
        }
    }
    print(f"[OT] dropTip: labware={labware_id}, well={well}, offset={off}")
    _post_command(cmd, wait=True)

def unload_to_trash(wellname: str = "A1", offset=None):
    """Drop the currently mounted tips into the fixed trash using dropTipInPlace."""
    if pipette_id is None:
        raise RuntimeError("Pipette not loaded.")

    off = offset or (0, 0, 0)
    print(f"[OT] unload_to_trash: fixedTrash, well={wellname}, offset={off}")

    # 1) Move above fixedTrash
    move_cmd = {
        "data": {
            "commandType": "moveToAddressableAreaForDropTip",
            "params": {
                "pipetteId": pipette_id,
                "addressableAreaName": "fixedTrash",
                "wellName": wellname,
                "wellLocation": {
                    "origin": "default",
                    "offset": {"x": off[0], "y": off[1], "z": off[2]},
                },
                "alternateDropLocation": False,
            },
            "intent": "setup",
        }
    }
    print("[OT] moveToAddressableAreaForDropTip -> fixedTrash")
    _post_command(move_cmd, wait=True)

    # 2) Eject
    drop_cmd = {
        "data": {
            "commandType": "dropTipInPlace",
            "params": {
                "pipetteId": pipette_id,
            },
            "intent": "setup",
        }
    }
    print("[OT] dropTipInPlace at fixedTrash")
    _post_command(drop_cmd, wait=True)

def pause_run(message="Tip check failed."):
    """Send a pause command."""
    cmd = {
        "data": {
            "commandType": "pause",
            "params": {"message": message},
            "intent": "protocol"
        }
    }
    print(f"[OT] pause: {message}")
    _post_command(cmd, wait=True)

def aspirate(volume, labware_id, wellname, offset=None, origin="bottom", flow_rate=150.0):
    """
    Aspirate liquid. Supports 'origin' ("top" or "bottom") and requires labware_id.
    """
    if pipette_id is None:
        raise RuntimeError("Pipette not loaded.")

    if labware_id is None or wellname is None:
        raise ValueError("aspirate() requires labware_id and wellname.")

    off = offset or (0, 0, 1)

    params = {
        "pipetteId": pipette_id,
        "volume": volume,
        "flowRate": flow_rate,
        "labwareId": labware_id,
        "wellName": wellname,
        "wellLocation": {
            "origin": origin, 
            "offset": {"x": off[0], "y": off[1], "z": off[2]},
        },
    }

    cmd = {
        "data": {
            "commandType": "aspirate",
            "params": params,
            "intent": "setup", # setup intent ensures immediate execution
        }
    }

    print(f"[OT] aspirate: {volume}uL from {wellname}, origin={origin}, offset={off}")
    _post_command(cmd, wait=True)

def dispense(volume, labware_id, wellname, offset=None, origin="bottom", flow_rate=150.0):
    """
    Dispense liquid. Supports 'origin' ("top" or "bottom") and requires labware_id.
    """
    if pipette_id is None:
        raise RuntimeError("Pipette not loaded.")

    if labware_id is None or wellname is None:
        raise ValueError("dispense() requires labware_id and wellname.")

    off = offset or (0, 0, 1)

    params = {
        "pipetteId": pipette_id,
        "volume": volume,
        "flowRate": flow_rate,
        "labwareId": labware_id,
        "wellName": wellname,
        "wellLocation": {
            "origin": origin,
            "offset": {"x": off[0], "y": off[1], "z": off[2]},
        },
    }

    cmd = {
        "data": {
            "commandType": "dispense",
            "params": params,
            "intent": "setup", # setup intent ensures immediate execution
        }
    }

    print(f"[OT] dispense: {volume}uL into {wellname}, origin={origin}, offset={off}")
    _post_command(cmd, wait=True)

def blow_out(labware_id=None, wellname=None, offset=None):
    """Blow out remaining liquid."""
    if pipette_id is None:
        raise RuntimeError("Pipette not loaded.")

    params = {
        "pipetteId": pipette_id,
        "flowRate": 100.0
    }

    if labware_id is not None or wellname is not None:
        if not (labware_id and wellname):
            raise ValueError("blow_out() needs both labware_id and wellname, or neither.")
        off = offset or (0, 0, 0)
        params.update({
            "labwareId": labware_id,
            "wellName": wellname,
            "wellLocation": {
                "origin": "top",  
                "offset": {"x": off[0], "y": off[1], "z": off[2]},
            },
        })

    cmd = {
        "data": {
            "commandType": "blowout",  
            "params": params,
            "intent": "setup", # Changed to setup for consistency
        }
    }
    print(f"[OT] blow_out")
    _post_command(cmd, wait=True)
    
def home():
    """Home the robot."""
    cmd = {
        "data": {
            "commandType": "home",
            "params": {},
            "intent": "setup"
        }
    }
    print("[OT] home: Returning to home position")
    _post_command(cmd, wait=True)