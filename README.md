# 🏭 AMD Multimodal Defect Agent

> **Submission for the AMD Developer Hackathon: ACT III** | *Intelligent Industry Track*

An AI-powered industrial copilot that combines real-time visual anomaly detection with dynamic maintenance log analysis to identify assembly line defects and recommend immediate root-cause corrective actions.

---

## 💡 Problem Statement

In modern high-speed manufacturing lines, detecting a surface defect is only half the battle. When an automated camera flags an anomaly, production line operators often face critical delays trying to determine **why** the defect occurred. 

Manual troubleshooting leads to prolonged operational downtime, increased scrap rates, and unnecessary maintenance overhead.

---

## 🚀 The Solution

The **AMD Multimodal Defect Agent** bridges the gap between computer vision inspection and operational intelligence:

1. **Visual Anomaly Detection:** Processes live part feeds against baseline reference images using OpenCV frame differential analysis to compute dynamic anomaly percentage scores.
2. **Contextual Root-Cause Retrieval:** Automatically queries local machine maintenance logs (`maintenance_logs.txt`) when anomaly scores cross configured safety thresholds.
3. **Agentic Copilot Action Plan:** Correlates visual defect patterns with recorded mechanical issues (e.g., belt vibration, guide misalignment) to deliver immediate, actionable recommendations to line operators.

---

## 🛠️ System Architecture & Tech Stack

```text
   +-----------------------+      +--------------------------+
   |   Live Image Feed     |      |  Historical Logs (RAG)   |
   +-----------+-----------+      +------------+-------------+
               |                               |
               v                               v
    [ OpenCV Differential ]        [ Maintenance Log Query ]
               |                               |
               +---------------+---------------+
                               |
                               v
               +-------------------------------+
               |    Streamlit Operator UI      |
               | (Visual Score + Diagnosis)    |
               +-------------------------------+
