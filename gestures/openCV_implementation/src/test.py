import os

# Define paths for the scripts
gesture_script_path = os.path.join(os.path.dirname(__file__), 'GestureRecognitionCVv2.py')
self_driving_script_path = os.path.join(os.path.dirname(__file__), 'SelfDriving.py')

# Display a message and prompt the user to choose
print("Welcome to the choice menu!")
print("1. Run Gesture Recognition Script")
print("2. Run Self Driving Script")

choice = input("Enter your choice (1 or 2): ").strip()

if choice == '1':
    print("Running GestureRecognitionCVv2.py...")
    # Execute GestureRecognitionCVv2.py
    with open(gesture_script_path) as file:
        exec(file.read())
elif choice == '2':
    print("Running SelfDriving.py...")
    # Execute SelfDriving.py
    with open(self_driving_script_path) as file:
        exec(file.read())
else:
    print("Invalid choice. Please enter 1 or 2.")
