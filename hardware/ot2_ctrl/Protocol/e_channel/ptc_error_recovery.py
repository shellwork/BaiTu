import sys
import os
from typing import List, Dict, Any

# Add path to access sibling modules if necessary
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def execute_recovery_plan(ot_module, recovery_tasks: List[Dict], context: Dict[str, Any]) -> bool:
    """
    Executes a list of recovery tasks based on the provided context.
    
    Args:
        ot_module: Reference to OT2_functions.
        recovery_tasks: List of task dictionaries (defined in config).
        context: Dictionary containing dynamic runtime data (run_id, wells, IDs, etc.).
        
    Returns:
        bool: True if all recovery tasks executed without crashing, False otherwise.
    """
    print(f"\n[RECOVERY] Starting Task-Based Recovery ({len(recovery_tasks)} steps)...")
    
    try:
        for i, task in enumerate(recovery_tasks):
            task_type = task.get('type')
            print(f"[RECOVERY] Step {i+1}: {task_type}")

            # --- Helper to resolve placeholders (e.g., 'ctx.source_well' -> 'A1') ---
            def resolve(key):
                if isinstance(key, str) and key.startswith("ctx."):
                    ctx_key = key.split("ctx.")[1]
                    val = context.get(ctx_key)
                    if val is None:
                        raise ValueError(f"Context key '{ctx_key}' missing in recovery context.")
                    return val
                return key

            # --- Dispatch Logic ---
            if task_type == 'dispense':
                ot_module.dispense(
                    volume=resolve(task.get('volume')),
                    labware_id=resolve(task.get('labware_id')),
                    wellname=resolve(task.get('well')),
                    origin="bottom"
                )

            elif task_type == 'blow_out':
                ot_module.blow_out(
                    labware_id=resolve(task.get('labware_id')),
                    wellname=resolve(task.get('well'))
                )

            elif task_type == 'drop':
                # Can drop to trash (default) or specific rack
                labware = resolve(task.get('labware_id')) # Can be None for fixedTrash
                ot_module.drop_tips(
                    tiprack_id=labware, 
                    wellname=resolve(task.get('well'))
                )

            elif task_type == 'home':
                ot_module.home()
            
            elif task_type == 'pause':
                ot_module.pause_run(message=task.get('message', "Paused by Recovery Protocol"))

            else:
                print(f"[RECOVERY][WARN] Unknown recovery task type: {task_type}")

        print("[RECOVERY] Plan completed successfully.\n")
        return True

    except Exception as e:
        print(f"[RECOVERY][CRITICAL] Recovery plan failed: {e}")
        # Emergency fallback: Try to home if everything else fails
        try:
            ot_module.home()
        except:
            pass
        return False