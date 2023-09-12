import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import plot3D

import networkx as nx
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic 

pose = mp_pose.Pose()

# Create a NetworkX graph to store the skeleton
G = nx.Graph()

fig = plt.figure()
axes = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

# Initialize webcam capture
cap = cv2.VideoCapture("C:/Users/lsf/Pictures/Camera Roll/WIN_20230912_16_10_26_Pro.mp4")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # Convert the frame to RGB
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Process the frame with MediaPipe Pose
  results = pose.process(frame_rgb)

  # If pose landmarks are detected, you can access them as results.pose_landmarks
  if results.pose_landmarks:
    landmarks_2d = results.pose_landmarks
    landmarks_3d = results.pose_world_landmarks
    
    mp_drawing.draw_landmarks(frame, landmarks_2d, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    plot3D.plot_world_landmarks(plt, axes, landmarks_3d)

     # Add nodes to the graph with 3D coordinates
    for idx, landmark in enumerate(landmarks_3d.landmark):
        v, x, y, z = landmark.visibility, landmark.x, landmark.y, landmark.z
        G.add_node(idx, visibility=v, x=x, y=y, z=z)
    
    # Add edges to the graph with the size stored as an attribute
    for edge in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = edge
        start_coords = np.array([G.nodes[start_idx]['x'], G.nodes[start_idx]['y'], G.nodes[start_idx]['z']])
        end_coords = np.array([G.nodes[end_idx]['x'], G.nodes[end_idx]['y'], G.nodes[end_idx]['z']])
        edge_size = np.linalg.norm(end_coords - start_coords)
        G.add_edge(start_idx, end_idx, size=edge_size)
    
    # Display the frame with landmarks
    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()