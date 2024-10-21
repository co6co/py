import subprocess
from co6co.utils.win import execute_command


def parse_netstat_output(output, resultIndex: int = -1) -> list[str] | None:
    result_processes = []
    if output == None:
        return None
    lines = output.strip().split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) > 0:
            result = parts[resultIndex]
            result_processes.append(result)
    return result_processes
