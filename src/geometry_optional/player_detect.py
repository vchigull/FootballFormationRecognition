from src.geometry_optional.player_detect import load_detector, detect_players
model, preprocess, device = load_detector(device='cpu') 
boxes, scores = detect_players(model, preprocess, 'path/to/frame.jpg', score_thresh=0.6)
