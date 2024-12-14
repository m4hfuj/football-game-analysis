from utils import read_video, write_video
from tracker.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner


def main():
    test_index = 2

    video_path = f"input_videos/test-{test_index}.mp4"

    frames = read_video(video_path)

    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(frames, read_from_stubs=True, stub_path=f"stubs/track_stubs-{test_index}.pkl")

    # Interpolate ball position
    tracks['balls'] = tracker.interpolate_ball_position(tracks['balls'])

 
    # Assign teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks["players"][frame_num][player_id]['team'] = team
            tracks["players"][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            # print(tracks['players'][frame_num][player_id])


    # Assign players to ball
    ball_assigner = PlayerBallAssigner()
    ## Team ball control
    team_ball_control = [0]
    for frame_num, player_tracks in enumerate(tracks["players"]):
        ball_bbox = tracks['balls'][frame_num][1]['bbox']
        assigned_player = ball_assigner.assign_player_to_ball(player_tracks, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            # print(tracks['players'][frame_num][assigned_player])
        else:
            team_ball_control.append(team_ball_control[-1])

    # print(team_ball_control)

    output_frames = tracker.draw_annotations(frames=frames, tracks=tracks, team_ball_control=team_ball_control)

    write_video(f"output_videos/out-{test_index}.avi", output_frames, fps=30)



if __name__ == '__main__':
    main()

