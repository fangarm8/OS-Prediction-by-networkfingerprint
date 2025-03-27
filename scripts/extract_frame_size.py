def extract_frame_size(sent_frames):
    if isinstance(sent_frames, list):
        for frame in sent_frames:
            if frame.get("frame_type") == "SETTINGS":
                return frame.get("settings", [])
    return None