import os
import sys

# --- CONFIGURACIÓN PARA LA CARPETA UNIFICADA DE WEBOTS ---
webots_home = os.environ.get("WEBOTS_HOME", "C:\\Program Files\\Webots")
webots_python_path = os.path.join(webots_home, "lib", "controller", "Python")

# Asegurarnos de que la ruta exista y añadirla a Python
if os.path.exists(webots_python_path) and webots_python_path not in sys.path:
    sys.path.append(webots_python_path)
# También aseguramos la ruta general de librerías de control
webots_controller_lib = os.path.join(webots_home, "lib", "controller")
if webots_controller_lib not in sys.path:
    sys.path.append(webots_controller_lib)
# ---------------------------------------------------------

from controller import Supervisor

# Inicializar el supervisor (que es el mismo Pioneer 3-AT)
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# Probar si podemos acceder a los nodos
robot_node = supervisor.getFromDef("ROBOT")
target_node = supervisor.getFromDef("TARGET")

if robot_node and target_node:
    print("✓ Setup successful!", flush=True)
    print(f"  Robot found at: {robot_node.getPosition()}", flush=True)
    print(f"  Target found at: {target_node.getPosition()}", flush=True)
else:
    print("✗ Setup error!", flush=True)