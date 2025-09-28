from crewai import Agent, Task, Crew
## Agents
# Fixed parse simulation info
def fixed_parse_simulation_info(input_text: str, agents_config: dict, tasks_config: dict):
    # Create agents
    location_parser = Agent(
        role=agents_config['location_parser_agent']['role'],
        goal=agents_config['location_parser_agent']['goal'],
        backstory=agents_config['location_parser_agent']['backstory'],
        verbose=agents_config['location_parser_agent']['verbose']
    )

    time_parser = Agent(
        role=agents_config['time_parser_agent']['role'],
        goal=agents_config['time_parser_agent']['goal'],
        backstory=agents_config['time_parser_agent']['backstory'],
        verbose=agents_config['time_parser_agent']['verbose']
    )

    # Create tasks with context as a list
    parse_location_task = Task(
        description=tasks_config['parse_location']['description'].format(input_text=input_text),
        expected_output=tasks_config['parse_location']['expected_output'],
        agent=location_parser
    )

    parse_time_task = Task(
        description=tasks_config['parse_time_period']['description'].format(input_text=input_text),
        expected_output=tasks_config['parse_time_period']['expected_output'],
        agent=time_parser
    )

    # Create crew
    crew = Crew(
        agents=[location_parser, time_parser],
        tasks=[parse_location_task, parse_time_task],
        verbose=True
    )

    # Run the crew
    result = crew.kickoff()
    
    # Extract the basin name from the output of the location parsing task, removing any surrounding quotes
    basin_name = parse_location_task.output.raw.strip('"\'')
    # Extract the time period from the output of the time parsing task (as a string, e.g., "[datetime(...)]")
    time_period = parse_time_task.output.raw

    # Return both pieces of information as a dictionary
    return {
        "basin_name": basin_name,
        "time_period": time_period
    }

def get_basin_center_coords(basin_name, input_text, agents_config: dict, tasks_config: dict):
    basin_center_agent = Agent(
        role=agents_config['basin_center_agent']['role'],
        goal=agents_config['basin_center_agent']['goal'],
        backstory=agents_config['basin_center_agent']['backstory'],
        verbose=agents_config['basin_center_agent']['verbose']
    )
    basin_center_task = Task(
        description=tasks_config['get_basin_center']['description'].format(basin_name=basin_name, input_text=input_text),
        expected_output=tasks_config['get_basin_center']['expected_output'],
        agent=basin_center_agent
    )
    crew = Crew(
        agents=[basin_center_agent],
        tasks=[basin_center_task],
        verbose=True
    )
    crew.kickoff()
    # Extract the output and try to parse the tuple using regex
    import re
    output = basin_center_task.output.raw
    match = re.search(r"\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?", output)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return (lat, lon)
    else:
        return output  # fallback: return raw output if parsing fails