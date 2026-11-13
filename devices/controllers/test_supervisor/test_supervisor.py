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