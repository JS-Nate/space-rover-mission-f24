        # Find the nearest target
        nearest_target = None
        nearest_target_label = None
        min_distance = float('inf')
        focused = False
        for center, label in target_centers:
            distance = np.linalg.norm(np.array(center) - np.array(bottom_object))
            if distance < min_distance:
                min_distance = distance
                nearest_target = center
                nearest_target_label = label


                # If we have a nearest target, track it over time
                if nearest_target:
                    # print(f"Nearest Target: {nearest_target_label.capitalize()}")
                    cv2.circle(frame, nearest_target, 5, (0, 255, 0), -1)
                    if nearest_target_start_time is None:
                        nearest_target_start_time = time.time()  # Start timer
                    elif time.time() - nearest_target_start_time >= 2:
                        if saved_target_position is None:
                            saved_target_position = nearest_target  # Save position after 2 sec
                else:
                    nearest_target_start_time = None  # Reset timer if no target found
                


                # Draw the saved red dot (if captured)
                if saved_target_position:
                    cv2.circle(frame, saved_target_position, 5, (0, 255, 0), -1)
                    # nearest_target = None  # Stop drawing the nearest target once saved_target_position is set

                    # Draw a red dot at the saved target position
                    target_dx = saved_target_position[0] - bottom_object[0]
                    target_dy = saved_target_position[1] - bottom_object[1]
                    target_angle_rad = np.arctan2(target_dy, target_dx)
                    target_angle_deg = np.degrees(target_angle_rad)

                    angle_diff = target_angle_deg - angle_deg
                    if angle_diff > 180:
                        angle_diff -= 360
                    elif angle_diff < -180:
                        angle_diff += 360

                    nearest_target_text = f"Nearest Target: {nearest_target_label.capitalize()}"
                    cv2.putText(frame, nearest_target_text, (20, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


                    cv2.line(frame, center_object, saved_target_position, (255, 0, 0), 2)

                    if center_object is not None and saved_target_position is not None:
                        line_length = np.linalg.norm(np.array(center_object) - np.array(saved_target_position))
                        length_text = f"Line Length: {line_length:.2f} pixels"
                    else:
                        line_length = 0  # or some default value

                    

                    length_text = f"Line Length: {line_length:.2f} pixels"
                    cv2.putText(frame, length_text, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                    # Display the angle difference on the screen
                    # angle_diff_text = f"Angle Diff: {angle_diff:.2f} degrees"
                    # cv2.putText(frame, angle_diff_text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


                    # Initialize a flag to remember if F has already been printed
                    f_printed = False

                    # Indicate on screen when the line is pointing at the nearest target
                    if abs(angle_diff) <= 10:
                        # If we're in focus mode and "Sent Message F" has not been printed yet
                        if line_length >= 60 and not f_printed:
                            # print("Sent Message F")
                            network_msg = "F"
                            cv2.putText(frame, f"Forward", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                            f_printed = True  # Set flag to indicate F has been printed

                        # Ensure the focus mode is activated
                        focused = True

                        # Update the focused text on screen
                        # focused_text = f"Focused: {focused}"

                        if line_length < 76:  # Threshold for "very small" distance
                            cv2.putText(frame, "Stop", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                            # Display the distance to the target on the screen
                            # print("Sent Message S")
                            network_msg = "S"

                            # Start a timer for 5 seconds to remove the current target
                            if nearest_target_label not in locked_targets:
                                locked_targets[nearest_target_label] = time.time()
                                # print(locked_targets)

                            # Check if 5 seconds have passed
                            if time.time() - locked_targets[nearest_target_label] >= 5:
                                # Remove the current target from detection
                                target_centers = [tc for tc in target_centers if tc[1] != nearest_target_label]
                                del locked_targets[nearest_target_label]
                                saved_target_position = None  # Reset saved target position
                                focused = False
                                f_printed = False  # Reset flag when focus mode ends
                            else:
                                # Ensure network_msg remains "S" during the 5 seconds
                                network_msg = "S"

                    else:
                        if focused == False:
                            if angle_diff > 0:
                                rotation_direction = "Right"
                                # print("Sent Message R")
                                network_msg = "R"
                            else:
                                rotation_direction = "Left"
                                # print("Sent Message L")
                                network_msg = "L"

                            # cv2.putText(annotated_frame, f"Rotate {rotation_direction}", (20, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

