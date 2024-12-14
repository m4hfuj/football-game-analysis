import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_distance

class PlayerBallAssigner:

    def __init__(self):
        self.max_distance_limit = 70


    def assign_player_to_ball(self, players_tracks, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, track in players_tracks.items():
            player_bbox = track['bbox']

            distance_left = get_distance( (player_bbox[0], player_bbox[3]), ball_position )
            distance_right = get_distance( (player_bbox[2], player_bbox[3]), ball_position )
            distance = min(distance_left, distance_right)

            if distance < minimum_distance:
                minimum_distance = distance
                if minimum_distance < self.max_distance_limit:
                    assigned_player = player_id

        return assigned_player


        






