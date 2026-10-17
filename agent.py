import os
import cv2
import numpy as np

# 1. Vision Detection Function (From Step 1)
def detect_anomaly(reference_path, target_path):
    ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    
    if ref is None or target is None:
        return {"is_defective": False, "error": "Could not load images."}
        
    ref = cv2.resize(ref, (500, 500))
    target = cv2.resize(target, (500, 500))
    
    diff = cv2.absdiff(ref, target)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    non_zero_count = np.count_nonzero(thresh)
    anomaly_score = (non_zero_count / (500 * 500)) * 100
    is_defective = anomaly_score > 1.5
    
    return {
        "is_defective": is_defective,
        "anomaly_score": round(anomaly_score, 2)
    }

# 2. Read Maintenance Logs (The Memory)
def load_maintenance_logs(filename="maintenance_logs.txt"):
    # Get the exact folder where agent.py lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check both the root folder and the data folder explicitly
    path_root = os.path.join(base_dir, filename)
    path_data = os.path.join(base_dir, "data", filename)
    
    for path in [path_root, path_data]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # Make sure the file isn't empty
                        return content
            except Exception:
                continue
                
    return "No maintenance logs found."
    
def run_root_cause_agent(machine_id, image_path):
    # Step A: Run vision check against a standard good reference
    vision_result = detect_anomaly("data/normal/sample_good.jpg", image_path)
    
    if not vision_result["is_defective"]:
        return f"Part looks good! Anomaly score: {vision_result['anomaly_score']}%"
    
    # Step B: Fetch logs if a defect is found
    logs = load_maintenance_logs()
    
    # Step C: Formulate the prompt/analysis 
    # (In a full setup, you pass this context to your local LLM on the AMD GPU)
    analysis_report = f"""
    [ALERT: DEFECT DETECTED ON {machine_id}]
    - Anomaly Score: {vision_result['anomaly_score']}%
    - Status: Defective Part Flagged on Line.
    
    [RETRIEVED HISTORICAL MAINTENANCE LOGS:]
    {logs}
    
    [AGENT HYPOTHESIS & RECOMMENDATION:]
    Based on the visual anomaly score and recent maintenance entries, 
    the surface defect on {machine_id} correlates with reported belt vibration/misalignment issues. 
    Recommended Action: Stop line temporarily, inspect mechanical alignment, and re-calibrate.
    """
    
    return analysis_report

# --- Test the Full Pipeline ---
if __name__ == "__main__":
    # Test with your defective image sample
    result_report = run_root_cause_agent("Machine-02", "data/defective/sample_defect.jpg")
    print(result_report)