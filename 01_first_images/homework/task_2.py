import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Определить номер полосы, на которую нужно перестроиться.

    :param image: изображение дороги с машинкой и препятствиями
    :return: номер полосы (0, 1, 2, ...) или -1, если перестраиваться не нужно
    """

    car_color_lower = np.array([0, 0, 150])  
    car_color_upper = np.array([100, 100, 255]) 
    obstacle_color_lower = np.array([150, 0, 0]) 
    obstacle_color_upper = np.array([255, 100, 100]) 

    car_mask = cv2.inRange(image, car_color_lower, car_color_upper)
    obstacle_mask = cv2.inRange(image, obstacle_color_lower, obstacle_color_upper)

    car_contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if car_contours:
        car_contour = max(car_contours, key=cv2.contourArea)
        car_moments = cv2.moments(car_contour)
        car_center = (int(car_moments['m10'] / car_moments['m00']), int(car_moments['m01'] / car_moments['m00']))

        num_lanes = len(obstacle_contours)+1

        lane_width = image.shape[1] // num_lanes
        road_number = car_center[0] // lane_width

   
        obstacle_in_current_lane = False
        for obstacle in obstacle_contours:
            obstacle_moments = cv2.moments(obstacle)
            if obstacle_moments['m00'] > 0:
                obstacle_center = (int(obstacle_moments['m10'] / obstacle_moments['m00']), int(obstacle_moments['m01'] / obstacle_moments['m00']))
                if (obstacle_center[0] // lane_width) == road_number:
                    obstacle_in_current_lane = True
                    break  

        if obstacle_in_current_lane:
            for new_road_number in range(num_lanes):
                obstacle_free = True
                for obstacle in obstacle_contours:
                    obstacle_moments = cv2.moments(obstacle)
                    if obstacle_moments['m00'] > 0:
                        obstacle_center = (int(obstacle_moments['m10'] / obstacle_moments['m00']), int(obstacle_moments['m01'] / obstacle_moments['m00']))
                        if (obstacle_center[0] // lane_width) == new_road_number:
                            obstacle_free = False
                            break 

                if obstacle_free:
                    return new_road_number

        return road_number 


    return -1