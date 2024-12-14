from ultralytics import YOLO
import supervision as sv
import pickle
import pandas as pd
import numpy as np
import os
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def interpolate_ball_position(self, ball_positions):
        ball_positions = [ x.get(1, {}).get('bbox', []) for x in ball_positions ]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [ {1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist() ]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.4)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stubs=False, stub_path=None):

        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            print(f"Loading track info stub from {stub_path}...")
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        # print(f"Loading track info stub from {stub_path}...")
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "balls": [],
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # Convert supervision to detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to players
            for object_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeepers":
                    detection_supervision.class_id[object_id] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["balls"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_names[cls_id] == "player":
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_names[cls_id] == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_names[cls_id] == "ball":
                    tracks["balls"][frame_num][1] = {"bbox": bbox}


        if stub_path is not None:
            print(f"Saving track info stub into {stub_path}...")
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        bbox_width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center = (x_center, y2),
            axes = (int(bbox_width), int(bbox_width*0.35)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness=2,
            lineType = cv2.LINE_4
        )

        rect_height = 20
        rect_width = 40
        x1_rect = int(x_center - rect_width//2)
        x2_rect = int(x_center + rect_width//2)
        y1_rect = int(y2 - rect_height//2 +15)
        y2_rect = int(y2 + rect_height//2 +15)

        if track_id is not None:
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color,
                cv2.FILLED
            )

            # x1_text = x1_rect + 12
            # if track_id > 99:
            track_id = int(track_id % 99)

            cv2.putText(
                frame,
                f"{str(track_id).zfill(2)}",
                (x1_rect+7, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    def draw_rectangle(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=(0, 0, 0),
            thickness=2
        )
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        team_1_frames = np.count_nonzero(team_ball_control_till_frame == 1)
        team_2_frames = np.count_nonzero(team_ball_control_till_frame == 2)

        if team_1_frames + team_2_frames:
            team_1 = team_1_frames / (team_1_frames+team_2_frames)
            team_2 = team_2_frames / (team_1_frames+team_2_frames)
            
            cv2.putText(frame, f"Team 1 ball control: {team_1*100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            cv2.putText(frame, f"Team 2 ball control: {team_2*100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annotations(self, frames, tracks, team_ball_control):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["balls"][frame_num]
            
            # Draw player annotations
            for track_id, player in player_dict.items():
                bbox = player['bbox']
                color = player.get('team_color', (0,0,0))
                self.draw_ellipse(frame, bbox, color, track_id)

                if player.get('has_ball', False):
                    self.draw_ellipse(frame, bbox, (0,0,255), track_id)

                # cv2.imwrite('output_videos/player.png', frame[ int(bbox[1]): int(bbox[3]), int(bbox[0]):int(bbox[2]) ] )


            # Draw referee annotations
            for track_id, referee in referee_dict.items():
                bbox = referee['bbox']
                self.draw_ellipse(frame, bbox, (0, 255, 255))

            # Draw rectangle on balls
            for track_id, ball in ball_dict.items():
                bbox = ball['bbox']
                # self.draw_rectangle(frame, bbox)

            # Draw team control
            team_ball_control = np.array(team_ball_control)
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_frames.append(frame)

        return output_frames


