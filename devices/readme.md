# Webots Reinforcement Learning Project (Pioneer 3-AT)

This repository contains a reinforcement learning setup using Webots and Python to control a Pioneer 3-AT robot moving towards a specific target.

---

## Prerequisites

* **Webots**: Make sure you have [Webots](https://cyberbotics.com/) installed on your system.
* **Python**: Python 3.10 installed.

---

## Installation & Setup

1. **Install Webots**
   Download and install the latest or compatible version of Webots from the [official website](https://cyberbotics.com/).

2. **Create a Virtual Environment**
   Set up a virtual environment using Python 3.10 and install the project dependencies:
   ```bash
   # Create virtual environment with Python 3.10
   python3.10 -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Install requirements
   pip install -r requirements.txt
3. **Open the Webots World**
    Open Webots and load the project world file located at:devices/worlds/MAS512.wbt _(It should be pre-configured.)_

4. **Start the Simulation**
    Press the Play button in Webots. You should see the following message in the Webots console:
    ```bash
    INFO: 'Pioneer 3-AT' extern controller: Waiting for local or remote connection on port 1234 targeting robot named 'Pioneer 3-AT' (Pioneer%203-AT)
5. **Open Your IDE**
    Open Visual Studio Code (or your preferred IDE) and ensure you activate the virtual environment you created in Step 2.

6. **Run the Controller Script**
    Execute the following command in your terminal:
    ```bash
    python controllers/test_supervisor/webots_train.py

_(Note: Ensure that PPO.load is configured to load the model file named PPO_webots_160k.)_

7. **Done!**
The robot will now start moving in the direction of the target.
