# PyBullet + Claude Vision Fetch Robot (`pybullet-claude-sim`)

An interactive 3D robotics simulation built with **PyBullet**, **FastAPI**, and **Anthropic Claude Vision** (`agent2.py`). A wheeled humanoid robot perceives the arena exclusively through its forward visor camera (with HUD bearing/floor-distance scales), a single center depth ray, and physical contact sensors to locate, pick up, and deliver objects into the target drop zone.

---

## Prerequisites

- **OS:** Windows 10 / 11 (or macOS / Linux with equivalent shell commands)
- **Python:** Python 3.11 (recommended for prebuilt `pybullet` wheel compatibility)
- **API Key:** An active Anthropic API key (`sk-ant-...`)

---

## 1. Clone & Set Up the Environment

Clone the repository and switch to the `pybullet-claude-sim` branch:

```cmd
git clone -b pybullet-claude-sim https://github.com/Roshni-07/HackOween_Codejuring.git
cd HackOween_Codejuring
```

Create and activate a virtual environment, then install dependencies:

```cmd
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Configure Your Anthropic API Key

Create a file named `api_key.txt` in the root folder (`HackOween_Codejuring/api_key.txt`) and paste your Anthropic API key inside it (single line, no quotes):

```cmd
echo sk-ant-your-api-key-here> api_key.txt
```

*(Note: `api_key.txt` is listed in `.gitignore` so it will never be committed to Git).*

---

## 3. Run the System (3 Terminals)

Open **three separate Command Prompt (`cmd`) terminals** inside the project directory (`HackOween_Codejuring`).

### Terminal 1: PyBullet Simulation & FastAPI Backend (`server.py`)
Starts the 3D PyBullet GUI arena and API server on `http://127.0.0.1:8000`:

```cmd
.\venv\Scripts\activate
uvicorn server:app --reload --port 8000
```

### Terminal 2: Claude Vision Agent (`agent2.py`)
Starts the interactive vision-language control loop:

```cmd
.\venv\Scripts\activate
python agent2.py
```

When prompted with `Command:`, enter a natural language instruction—for example:
- `Pick up the orange block and drop it in the pumpkin zone`
- `Pick up the green cube and drop it in the pumpkin zone`
- `Fetch the red sphere and place it in the pumpkin zone`

**Interactive Controls in Terminal 2:**
- After completing a task, `agent2.py` automatically prompts for your next command.
- Type `reset` to return the robot to the center `(0, 0, 0)` and re-scatter the arena objects.
- Type `exit` or `quit` to close the agent.

### Terminal 3: Web Control & Live Visor Dashboard (`index.html`)
Serves the browser dashboard on `http://127.0.0.1:5500`:

```cmd
python -m http.server 5500
```

Open **`http://127.0.0.1:5500/index.html`** in your browser to view the live **Visor Camera**, **Live Telemetry**, and manual WASD/Grab/Drop controls.

---

## Useful CLI Flags for `agent2.py`

- **Test API key validity (free, 0 tokens):**
  ```cmd
  python agent2.py --check-key
  ```
- **Test robot movement & camera plumbing without API calls:**
  ```cmd
  python agent2.py --dummy
  ```
- **Verify server movement and camera frame updates:**
  ```cmd
  python agent2.py --selftest
  ```
- **Use a faster/cheaper Claude model:**
  ```cmd
  python agent2.py --model claude-haiku-5-5
  ```

---

## Repository Structure

- `server.py` — FastAPI server exposing `/look`, `/depth`, `/state`, `/move`, `/turn`, `/pickup`, `/drop`, and `/reset`.
- `sim_engine.py` — PyBullet physics simulation engine, humanoid kinematics, and held-object + torso collision detection.
- `sim/` — Modular simulation subsystems (`scene.py`, `driver.py`, `sensors.py`, `referee.py`).
- `agent2.py` — Interactive Claude Vision agent with HUD overlay, odometry memory, and continuous command loop.
- `index.html` — Live visor camera stream, telemetry viewer, and manual teleoperation dashboard.
