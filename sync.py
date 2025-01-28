from neo4j import GraphDatabase
import csv

# Neo4j connection settings
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"

# Function to create nodes and relationships
def load_data_to_neo4j(csv_file):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        # Clear existing data (optional)
        session.run("MATCH (n) DETACH DELETE n")

        # Read the CSV and upload data
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Create Employee node
                session.run(
                    """
                    MERGE (e:Employee {name: $name})
                    SET e.gender = $gender,
                        e.startDate = $start_date,
                        e.lastLoginTime = $last_login,
                        e.salary = $salary,
                        e.bonus = $bonus,
                        e.seniorManagement = $senior_management
                    """,
                    name=row["First Name"],
                    gender=row["Gender"],
                    start_date=row["Start Date"],
                    last_login=row["Last Login Time"],
                    salary=float(row["Salary"]),
                    bonus=float(row["Bonus %"]),
                    senior_management=row["Senior Management"] == "TRUE"
                )

                # Create Team node
                session.run(
                    """
                    MERGE (t:Team {name: $team_name})
                    """,
                    team_name=row["Team"]
                )

                # Create BELONGS_TO relationship
                session.run(
                    """
                    MATCH (e:Employee {name: $name}), (t:Team {name: $team_name})
                    MERGE (e)-[:BELONGS_TO]->(t)
                    """,
                    name=row["First Name"],
                    team_name=row["Team"]
                )

    driver.close()

# Usage
csv_file_path = "employees.csv"  # Replace with your CSV file path
load_data_to_neo4j(csv_file_path)

