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

fig1 = plt.figure(1)
# fig2 = plt.figure(2, figsize=(10,6))
axes = fig1.add_subplot(111, projection="3d")
fig1.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

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

  # class PoseLandmark(enum.IntEnum):
    #   """The 33 pose landmarks."""
    #   NOSE = 0
    #   LEFT_EYE_INNER = 1
    #   LEFT_EYE = 2
    #   LEFT_EYE_OUTER = 3
    #   RIGHT_EYE_INNER = 4
    #   RIGHT_EYE = 5
    #   RIGHT_EYE_OUTER = 6
    #   LEFT_EAR = 7
    #   RIGHT_EAR = 8
    #   MOUTH_LEFT = 9
    #   MOUTH_RIGHT = 10
    #   LEFT_SHOULDER = 11
    #   RIGHT_SHOULDER = 12
    #   LEFT_ELBOW = 13
    #   RIGHT_ELBOW = 14
    #   LEFT_WRIST = 15
    #   RIGHT_WRIST = 16
    #   LEFT_PINKY = 17
    #   RIGHT_PINKY = 18
    #   LEFT_INDEX = 19
    #   RIGHT_INDEX = 20
    #   LEFT_THUMB = 21
    #   RIGHT_THUMB = 22
    #   LEFT_HIP = 23
    #   RIGHT_HIP = 24
    #   LEFT_KNEE = 25
    #   RIGHT_KNEE = 26
    #   LEFT_ANKLE = 27
    #   RIGHT_ANKLE = 28
    #   LEFT_HEEL = 29
    #   RIGHT_HEEL = 30
    #   LEFT_FOOT_INDEX = 31
    #   RIGHT_FOOT_INDEX = 32

  # If pose landmarks are detected, you can access them as results.pose_landmarks
  if results.pose_landmarks:
    landmarks_2d = results.pose_landmarks
    landmarks_3d = results.pose_world_landmarks
    
    mp_drawing.draw_landmarks(frame, landmarks_2d, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    plt.figure(1)
    plot3D.plot_world_landmarks(plt, axes, landmarks_3d)

    # Add nodes to the graph with 3D coordinates
    for idx, landmark in enumerate(landmarks_3d.landmark):
        v, x, y, z = landmark.visibility, landmark.x, landmark.y, landmark.z
        G.add_node(idx, visibility=v, x=x, y=y, z=z)

    # Add edges to the graph with the size stored as an attribute
    for bone in mp_pose.POSE_CONNECTIONS:
      start_idx, end_idx = bone
      start_coords = np.array([G.nodes[start_idx]['x'], G.nodes[start_idx]['y'], G.nodes[start_idx]['z']])
      end_coords = np.array([G.nodes[end_idx]['x'], G.nodes[end_idx]['y'], G.nodes[end_idx]['z']])
      bone_vector = end_coords - start_coords
      bone_size = np.linalg.norm(bone_vector)

      # Calculate fuzzy-like features
      is_in_front_of = np.clip((end_coords[2] - start_coords[2]) / bone_size, -1, 1)
      is_above = np.clip((end_coords[1] - start_coords[1]) / bone_size, -1, 1)
      is_rightwards = np.clip((end_coords[0] - start_coords[0]) / bone_size, -1, 1)
      
      # Add edge attributes
      G.add_edge(
        start_idx, 
        end_idx,
        vector=bone_vector, 
        size=bone_size, 
        in_front_of=is_in_front_of, 
        is_above=is_above, 
        is_rightwards=is_rightwards)
    
    # # Add edges to connect all landmarks
    # num_landmarks = len(landmarks_3d.landmark)
    # for i in range(num_landmarks):
    #   for j in range(i+1, num_landmarks):
    #     if not G.has_edge(i, j):
    #       start_coords = np.array([G.nodes[i]['x'], G.nodes[i]['y'], G.nodes[i]['z']])
    #       end_coords = np.array([G.nodes[j]['x'], G.nodes[j]['y'], G.nodes[j]['z']])
    #       bone_vector = end_coords - start_coords
    #       bone_size = np.linalg.norm(bone_vector)
          
    #       # Calculate fuzzy-like features for the new edges
    #       is_in_front_of = np.clip((end_coords[2] - start_coords[2]) / bone_size, -1, 1)
    #       is_above = np.clip((end_coords[1] - start_coords[1]) / bone_size, -1, 1)
    #       is_rightwards = np.clip((end_coords[0] - start_coords[0]) / bone_size, -1, 1)
          
    #       # Add edge attributes for the new edges
    #       G.add_edge(i, j, size=bone_size, in_front_of=is_in_front_of, above=is_above, rightwards=is_rightwards)
    
    # plt.figure(2).clear()
    # pos = nx.spring_layout(G, k=8)
    # nx.draw(G, pos , with_labels = True, width=0.4, 
    #         node_color='lightblue', node_size=400)
    
    # Display the frame with landmarks
    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()