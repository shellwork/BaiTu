"""
8-Channel Pipette Protocol with Vision Check
Pick tip -> Move to imaging position -> Capture image -> Model check -> Continue or Pause
"""

import os
import sys

# Add paths so we can import OT2_functions and detection_functions
current_dir = os.path.dirname(os.path.abspath(__file__))
ot2_ctrl_dir = os.path.join(current_dir, '../..')
as_project_dir = os.path.join(current_dir, '../../..')
ptc_utils_dir = os.path.join(current_dir, '../')

sys.path.insert(0, ot2_ctrl_dir)
sys.path.insert(0, as_project_dir)
sys.path.insert(0, ptc_utils_dir)

import OT2_functions as OT
import detection_functions as Vision
import ptc_utils as utils

from typing import List, Dict, Tuple


from IPython.display import display
from PIL import Image as PILImage
import matplotlib.pyplot as plt

"""
NEW Task Based Configuration Portocal Design
"""

import os
import sys
from typing import List, Dict, Any

# Adjust paths as per your project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '../..')) # OT2_functions path
sys.path.insert(0, current_dir)

import OT2_functions as OT
import detection_functions as Vision
import ptc_error_recovery as ErrorRecovery
import ptc_utils as utils


def _get_default_liquid_recovery():
    """Returns a default recovery list if none is provided in config."""
    return [
        {"type": "dispense", "labware_id": "ctx.source_labware_id", "well": "ctx.source_well", "volume": "ctx.volume"},
        {"type": "blow_out", "labware_id": "ctx.source_labware_id", "well": "ctx.source_well"},
        {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.pick_well"},
        {"type": "home"}
    ]

def _get_default_pickup_recovery():
    """Returns a default recovery list if none is provided in config."""
    return [
        {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.well_name"},
        {"type": "home"}
    ]


def execute_protocol(config: Dict[str, Any], model: Any):
    """
    Main executor for Task-Based Configuration.
    
    Args:
        config: Dictionary containing 'labware' and 'tasks'.
        model: Loaded YOLO model.
    """
    print("\n" + "=" * 60)
    print("  STARTING TASK-BASED PROTOCOL")
    print("=" * 60 + "\n")

    # 1. Setup Phase: Create Run and Load Labware
    run_id, _ = OT.create_run()
    print(f"[EXECUTOR] Run ID: {run_id}")
    
    # Load labware using your existing util
    labware_map = utils.load_labware_from_config(OT, config['labware'])
    
    # Extract IDs for easy access
    pipette_id = labware_map['pipette_id']
    tiprack_id = labware_map['tiprack_id']
    img_labware_id = labware_map['imaging_labware_id']
    img_well = config['settings']['imaging_well']
    img_offset = config['settings'].get('imaging_offset', (0, 0, 50))
    base_dir = config['settings'].get('base_dir', 'Test_Task')
    
    # State tracking
    current_tip_well = None 
    
    # 2. Task Execution Loop
    tasks = config.get('tasks', [])
    
    for i, task in enumerate(tasks):
        task_type = task.get('type')
        print(f"\n[EXECUTOR] Processing Task {i+1}: {task_type.upper()}")
        
        if task_type == 'pickup':
            success = _execute_pickup(
                task, run_id, model, tiprack_id, img_labware_id, 
                img_well, img_offset, base_dir
            )
            if success:
                current_tip_well = task['well']
            else:
                print("[EXECUTOR] Protocol stopping due to Pickup Failure.")
                break
                
        elif task_type == 'transfer':
            # Resolve Source/Dest IDs from the labware_map
            # The config should use slot names or keys to reference labware
            src_slot = task['source_slot']
            dest_slot = task['dest_slot']
            
            # Handle source labware (account for multiple sources logic in utils)
            src_id = labware_map['sources'][src_slot]
            
            # Handle dispense labware (account for single/multi logic)
            if 'dispenses' in labware_map:
                dest_id = labware_map['dispenses'][dest_slot]
            else:
                dest_id = labware_map['dispense_labware_id']

            success = _execute_transfer(
                task, run_id, model, src_id, dest_id, img_labware_id,
                img_well, img_offset, base_dir,
                # Context for recovery
                tiprack_id=tiprack_id,
                current_tip_well=current_tip_well
            )
            if not success:
                print("[EXECUTOR] Protocol stopping due to Transfer Failure.")
                break

        elif task_type == 'drop':
            OT.drop_tips(tiprack_id=tiprack_id, wellname=task['well'])
            current_tip_well = None
            print("[EXECUTOR] Tips dropped.")

    print("\n[EXECUTOR] Protocol Execution Finished.")
    OT.home()


def _execute_pickup(task, run_id, model, tiprack_id, img_id, img_well, img_offset, base_dir):
    """Helper to execute pickup and vision check."""
    well = task['well']
    
    # 1. Action
    OT.pick_up(tiprack_id=tiprack_id, wellname=well)
    
    # 2. Vision Check (Standardized Flow)
    
    # Capture & Predict
    result = Vision.Predict(
        ot_module=OT,
        model=model,
        run_id=run_id,
        check_type='pickup',
        imaging_labware_id=img_id,
        imaging_well=img_well,
        imaging_offset=img_offset,
        base_dir=base_dir,
        step_name=f"pickup_{well}",
        # Task specific args
        conf=task.get('conf', 0.6),
        expected_tips=task.get('expected_tips', 8)
    )
    
    # 3. Logic
    if result['passed']:
        print("[EXECUTOR] Tip Pickup Verified.")
        return True
    else:
        print(f"[RECOVERY REASON] Missing: {result.get('missing_positions')}, Presence: {result.get('tip_presence')}")
        recovery_context = {
            "run_id": run_id,
            "pipette_id": OT.pipette_id, 
            "tiprack_id": tiprack_id,
            "well_name": well,   # Pickup 只有 'well_name' (即 target well)
            "vision_result": result
        }

        # 5. Get Recovery Plan from Config (or use default)
        recovery_plan = task.get('on_fail', _get_default_pickup_recovery())        
        # 6. Execute Recovery
        ErrorRecovery.execute_recovery_plan(OT, recovery_plan, recovery_context)
        return False


def _execute_transfer(task, run_id, model, src_id, dest_id, img_id, img_well, img_offset, base_dir, tiprack_id, current_tip_well):
    """Helper to execute Aspirate -> Check -> Dispense."""
    vol = task['volume']
    src_well = task['source_well']
    dest_well = task['dest_well']
    
    # 1. Aspirate
    OT.aspirate(
        volume=vol, 
        labware_id=src_id, 
        wellname=src_well, 
        origin=task.get('origin', 'top'),
        offset=task.get('offset', (0,0,-35)) # Deep well default
    )
    
    # 2. Vision Check
    OT.move(img_id, wellname=img_well, offset=img_offset)
    
    result = Vision.Predict(
            ot_module=OT,
            model=model,
            run_id=run_id,
            check_type='transfer',
            imaging_labware_id=img_id,
            imaging_well=img_well,
            imaging_offset=img_offset,
            base_dir=base_dir,
            step_name=f"asp_{src_well}",
            # Task specific args
            conf=task.get('conf', 0.6),
            volume=vol,
            expected_tips=task.get('expected_tips', 8)
        )
    print(f"[EXECUTOR] Liquid Level Check Result: {result['passed']}")
    # 3. Logic
    if result['passed']:
        print("[EXECUTOR] Liquid Level Verified. Proceeding to Dispense.")
        OT.dispense(
            volume=vol,
            labware_id=dest_id,
            wellname=dest_well,
            origin="bottom"
        )
        OT.blow_out(labware_id=dest_id, wellname=dest_well)
        return True
    else:
        # 4. Failure Handler
        print(f"[RECOVERY REASON]{result['detected_levels']}, {result['expected_height_percent']}")
        recovery_context = {
            "run_id": run_id,
            "pipette_id": OT.pipette_id, # Access global from OT module
            "source_labware_id": src_id,
            "source_well": src_well,
            "dest_labware_id": dest_id,
            "dest_well": dest_well,
            "tiprack_id": tiprack_id,
            "pick_well": current_tip_well,
            "volume": vol,
            "vision_result": result
        }

        # 5. Get Recovery Plan from Config (or use default)
        recovery_plan = task.get('on_fail', _get_default_liquid_recovery())
        
        # 6. Execute Recovery
        ErrorRecovery.execute_recovery_plan(OT, recovery_plan, recovery_context)
        return False # Stop protocol after recovery
        

def main():
    """Example usage placeholder (model loading should happen outside)."""
    # from super_gradients.training import models
    # model_path = os.path.join(as_project_dir, "Model_weight/average_model.pth")
    # model = models.get("yolo_nas_l", num_classes=2, checkpoint_path=model_path)
    #
    # result = run_tip_pickup_with_vision_check(model=model)
    # print(result)

    print("New PTC module imported successfully!")
    pass


if __name__ == "__main__":
    main()
