import numpy as np 

# Function to calculate Euclidean distance between keypoints
def calculate_distance(keypoint1, keypoint2):
    return np.linalg.norm(keypoint1 - keypoint2)

# Function to calculate PCK
def calculate_pck(predicted_keypoints, ground_truth_keypoints, threshold):
    num_correct = 0
    total_keypoints = len(predicted_keypoints)
    
    for pred_kp, gt_kp in zip(predicted_keypoints, ground_truth_keypoints):
        distance = calculate_distance(pred_kp, gt_kp)
        if distance <= threshold:
            num_correct += 1
    
    pck = (num_correct / total_keypoints) * 100
    return pck