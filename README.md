
# ![Demo GIF](assets/football.png) Football Game Analysis Pipeline

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/-YOLOv8-FF9900?logo=yolo&logoColor=white)
![ByteTrack](https://img.shields.io/badge/-ByteTrack-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
---

This repository contains a robust pipeline for analyzing football game videos, focusing on object detection, tracking, team assignments, and ball control analysis. The program leverages advanced computer vision techniques to process game footage, delivering detailed insights through annotations and statistics.

![Demo GIF](assets/demo.gif)
---

## 🚀 Features

### 🔍 Object Tracking
- Detects players, referees, and the ball using a **YOLO** model.
- Tracks objects across video frames with **ByteTrack**.
- Smooths missing ball positions using interpolation.

### ⚙️ Team Assignment
- Uses **KMeans clustering** on player jersey colors to classify players into two teams.
- Assigns team colors to players and updates their roles dynamically.

### 🏐 Ball Ownership
- Identifies which player has possession of the ball.
- Tracks ball control transitions across frames.
- Calculates ball possession percentages for each team.

### 🎥 Visualization
- Annotates video frames with:
  - Bounding boxes
  - Team colors
  - Player IDs
  - Ball control statistics
- Outputs a fully annotated video.

---



## 📂 Project Structure

```plaintext
football-game-analysis/
├── data/
│   ├── input_videos/        # Raw football game videos
│   ├── output_videos/       # Processed videos with annotations
├── src/
│   ├── tracker.py           # Object tracking with YOLO and ByteTrack
│   ├── team_assigner.py     # KMeans clustering for team assignments
│   ├── ball_control.py      # Player-ball assignment and control tracking
│   ├── visualize.py         # Frame annotation scripts
│   ├── main.py              # Main pipeline script
├── tests/
│   ├── test_tracker.py      # Unit tests for tracker module
│   ├── test_team_assigner.py# Unit tests for team assignment
├── requirements.txt         # Python dependencies
├── README.md                # Documentation
└── LICENSE                  # License file
