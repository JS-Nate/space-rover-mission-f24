# Open Liberty Space Rover Mission

Open Liberty Space Rover Mission is an interactive IoT game designed to showcase microservices with modern Java technologies in a fun way. The mission is to guide the Space Rover using hand gestures to its home while visiting planets along the way to gather supplies. Make sure to avoid any obstacles (asteroids, black-hole, etc.) in your path. 🚨 Beware of extra damage on the Sun! 🚨

![Teaser](/images/teaser.jpg)

### Five Hand Gestures
These five hand gestures are used to move the Space Rover across the game board. <img width="692" alt="image" src="https://user-images.githubusercontent.com/45052124/173872753-3410f9c0-42ac-49f4-a52a-ebbd8ff6ee70.png">


The gesture detection is built upon using popular open sourced computer vision libraries such as [OpenCV](https://github.com/opencv/opencv-python), [CVZone](https://github.com/cvzone/cvzone) and [Mediapipe](https://github.com/google/mediapipe). Computer Vision works by gaining high-level understanding from digital images/videos to understand and automate tasks that the human visual system can do. OpenCV was built to provide a common infrastructure for Computer Vision applications and to accelerate the use of machine perception.

## Prerequisites
1. [Space Rover hardware component](https://github.com/OpenLiberty/space-rover-mission/tree/main/devices/space-rover) with the project code compiled and running.
2. [Game Board hardware component](https://github.com/OpenLiberty/space-rover-mission/tree/main/devices/game-board) with the project code compiled and running.
3. [Gesture Control System client](https://github.com/OpenLiberty/space-rover-mission/tree/main/gestures) with webcam facing a defined area for users to control the Space Rover.
4. Docker CE or [Docker Desktop](https://www.docker.com/products/docker-desktop/) either running on the Gesture Control System's hardware or on its own hardware. 
5. Kubernetes

## Setup, Cleanup, and Troubleshooting
For setup, cleanup, and troubleshooting instructions, see [documentation](https://github.com/OpenLiberty/space-rover-mission/tree/main/documentation).

![GameBoard](/images/gameboard-v2.png)

## Gameplay

Open Liberty Space Rover Mission is playable in four different game modes: Classic, Planet Hop, Guided, and Sudden Death.

1. **Classic**: Pick up all four planet resources (in any desired order) as fast as possible while avoiding obstacles.
2. **Planet Hop**: One planet will randomly turn on at a time -- go to the planet that is lit up to retrieve its resource. The availability of the planet's resource runs on a timer -- you can only retrieve the resource when its light is on! The planet's light will start blinking if you are running low on time to retrieve its resource, and the next planet's light will automatically turn on when the time is up. Pick up as many planet resources as possible within the time limit while avoiding obstacles.
3. **Guided**: One planet will turn on at a time in a specific order. Pick up all four planet resources in the order that the planet lights turn on as fast as possible, while avoiding obstacles.
4. **Sudden Death**: A variant of **Classic** mode, but if you hit an obstacle, you immediately fail the mission.
5. **Tutorial Mode**: Uses YOLO to detect planets and drive itself

## Architecture Diagram

![ArchitectureDiagram](/images/architecture.png)

#### Container 1 - Client
The Client is built using React and served using Nginx. The UI is used to enter the player's name, start the game, view current game stats, and view the leaderboard. It interacts with the Game Service using websockets and the Leaderboard Service using HTTP.

#### Container 2 - Leaderboard Serivce
The Leaderboard Service is built using Open Liberty. It is used to provide a REST API for the Client to query for past game information and for the Game Service to add the information after a game completes.

#### Container 3 - Game Service
The Game Service is built using Open Liberty. It holds the current game state and is the piece that connects all the different components together. It gets information from the hardware devices using websockets and sends events to the Client to update the health and score. It also updates the Leaderboard Service about the game information at the end of a game.

#### Container 4 - MongoDB
MongoDB is used to store information about past games. It interacts with the Leaderboard Service.

#### Container 5 - Prometheus server
The Prometheus server scrapes metrics from the Game Service and sends them to Grafana.

#### Container 6 - Grafana
Grafana takes the metrics scraped by the Prometheus Server and displays them on a dashboard.

#### Webcam and WiFi Enabled Standalone Device
The Webcam and WiFi Enabled Standalone Device is used to run the hand gesture game controls which is powered by CV Zone which uses Open CV under the hood.

#### Space Rover
The Space Rover is the physical device built using an Arduino which roams the game map. This component picks up RFID cards/NFC stickers to determine if it has landed on a planet or hit an obstacle to the Game Service.

#### Game Map with Addressable LED's
The Game Map is the physical game board which the Space Rover roams on. It emits LED's of different colour to signify planets and obstacles.

## Jakarta EE and MicroProfile features

### Docker Container 2 – Leaderboard service

#### Jakarta EE 9.1
- [RESTful Web Services (JAX-RS) 3.0](https://jakarta.ee/specifications/restful-ws/)
    -	Used for REST endpoints to interface with the leaderboard service.
    -	HTTP GET endpoint used to retrieve stats from MongoDB container.
    -	HTTP POST to update MongoDB with new statistics.
- [Context Dependency Injection (CDI) 3.0](https://jakarta.ee/specifications/cdi/3.0/)
    - Used to inject a MongoDatabase object and MongoClient used to interact with the MongoDB leaderboard database.

#### MicroProfile 5.0
- [MP Health 4.0](https://download.eclipse.org/microprofile/microprofile-health-4.0/microprofile-health-spec-4.0.html)
    -	Used to add a /health endpoint that determines if MongoDB is up and running.
- [MP Config 3.0](https://download.eclipse.org/microprofile/microprofile-config-3.0/microprofile-config-spec-3.0.html)
    - Used to store connection information to MongoDB such as the username, port, and encrypted password.
- [MP FaultTolerance 4.0](https://download.eclipse.org/microprofile/microprofile-fault-tolerance-4.0/microprofile-fault-tolerance-spec-4.0.html)
	- Used to retry connection attempts to MongoDB if they are not successful. 
- [MP OpenAPI 3.0](https://download.eclipse.org/microprofile/microprofile-open-api-3.0/microprofile-openapi-spec-3.0.html)
	- Used for providing REST API documentation and UI for demonstration.

### Docker Container 3 – Game service

#### Jakarta EE 9.1
- [WebSocket 2.0](https://jakarta.ee/specifications/websocket/)
    - Used extensively to handle socket connections from the GUI, Gesture Control Service, Game Board, and Space Rover.

#### MicroProfile 5.0
- [MP Health 4.0](https://download.eclipse.org/microprofile/microprofile-health-4.0/microprofile-health-spec-4.0.html)
    - Used to add a /health endpoint which returns the server status based on its ability to handle messages within a timeout.
- [MP Config 3.0](https://download.eclipse.org/microprofile/microprofile-config-3.0/microprofile-config-spec-3.0.html)
    - Used to configure known IP addresses and ports for external connections to the Space Rover, Game Board and Leaderboard.
- [MP Metrics 3.0](https://download.eclipse.org/microprofile/microprofile-metrics-3.0/microprofile-metrics-spec-3.0.html)
    - Used to record JVM metrics on the game service as well as a SimpleTimer that tracks the amount of time it takes for the GameService to respond to the health check test. This tells us a local round trip latency value determining how long it takes a message to be sent to the server and have the server respond.
- [MP Rest Client 3.0](https://download.eclipse.org/microprofile/microprofile-rest-client-3.0/microprofile-rest-client-spec-3.0.html)
    - Generates an HTTP client to send game end statistics to the leaderboard's REST server.
- [MP OpenAPI 3.0](https://download.eclipse.org/microprofile/microprofile-open-api-3.0/microprofile-openapi-spec-3.0.html)
	- Used for providing REST API documentation and UI for demonstration.

[![Gitter](https://badges.gitter.im/YourGitterRoom/community.svg)](https://gitter.im/openliberty-spacerover/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


## 2024-2025 additions

### Self Driving
-Includes a tutorial mode for beginner players 
![image](https://github.com/user-attachments/assets/1662cc64-472c-4ed0-bc4c-21a187a4595d)

-Uses the YOLO algorithim for object detection 
-Resorted it to a GPU accelerated mode (through NVIDIA) for faster response time 
![image](https://github.com/user-attachments/assets/8bc03597-e7f6-4c95-b121-1cbe4b05c168)



### Gameboard 
-Constructed fencing to safekeep Rover
-Printed matte poster
-Object layout of planets and details

### Network
-used LED ring light display to show the battery perecentage to prevent shutdowns
-Color code: 100% = green, 50% = yellow, <20% = red
![image](https://github.com/user-attachments/assets/9e6e2c4e-a901-4e84-9786-1a62bf29a955)


### Grafana Dashboard
-Detects the Rover's temprature, battery, power level to prevent overheating and make user aware of system state
![image](https://github.com/user-attachments/assets/daab6603-f865-4f25-9a79-5273812cd5c5)


### Steering
-Built ball castor under Rover to prevent wear and tear
![image](https://github.com/user-attachments/assets/9b0d0b63-a213-4ed4-ae80-9e06ebeb1e2f)


Note: In the length of this project we had the privelge of resorting to many resources for assitance, one of them being ChatGPT. This resource was used to debug code, clean up lengthy unnecessary lines, research YOLO algorithm techniques, calculate simple math calculations (distance and proximity), develop tables, and validate ideas with advancing cloud based systems.




