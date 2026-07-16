def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """

    task_map = {task["name"]: task.copy() for task in tasks}

    waiting = {}
    ready = {}
    running = {}
    completed = set()

    output = []
    current_resources = 0
    time = 0

    # Initialize waiting and ready queues
    for task in task_map.values():
        task["remaining_time"] = task["duration"]

        if task["depends_on"]:
            waiting[task["name"]] = task
        else:
            ready[task["name"]] = task

    while len(completed) < len(tasks):

        # Start ready tasks (alphabetical order)
        for name in sorted(list(ready.keys())):
            task = ready[name]

            if current_resources + task["resources"] <= resource_budget:
                running[name] = task
                current_resources += task["resources"]
                output.append((name, time))
                del ready[name]

        # Advance one unit of time
        time += 1

        # Update running tasks
        finished = []

        for name, task in running.items():
            task["remaining_time"] -= 1
            if task["remaining_time"] == 0:
                finished.append(name)

        # Finish completed tasks
        for name in finished:
            task = running.pop(name)
            current_resources -= task["resources"]
            completed.add(name)

        # Move newly unblocked tasks to ready
        unlocked = []

        for name, task in waiting.items():
            if all(dep in completed for dep in task["depends_on"]):
                unlocked.append(name)

        for name in unlocked:
            ready[name] = waiting.pop(name)

    return output