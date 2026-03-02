import sys
import traceback

def run_script():
    try:
        with open("output_log.txt", "w") as f:
            f.write("Starting...\n")
        # Just try to import to see if there are any import errors
        import propabridge_vertexai_pipeline
        import agent_orchestrator
        with open("output_log.txt", "a") as f:
            f.write("Imports successful.\n")
    except Exception as e:
        with open("output_log.txt", "a") as f:
            f.write(traceback.format_exc())

run_script()
