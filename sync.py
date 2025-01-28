import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"  # Replace with your password

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        with self.driver.session() as session:
            # Create constraints for unique IDs
            try:
                session.run("CREATE CONSTRAINT employee_id IF NOT EXISTS FOR (e:Employee) REQUIRE e.id IS UNIQUE")
                session.run("CREATE CONSTRAINT department_name IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE")
            except Exception as e:
                print(f"Constraint creation warning (might already exist): {e}")

    def create_employee(self, employee_data, employee_id):
        with self.driver.session() as session:
            # Create employee node
            cypher_query = """
            MERGE (e:Employee {id: $id})
            SET e.name = $name,
                e.gender = $gender,
                e.start_date = $start_date,
                e.last_login = $last_login,
                e.salary = $salary,
                e.bonus = $bonus,
                e.is_senior_management = $is_senior_management,
                e.team = $team
            WITH e
            MERGE (d:Department {name: $department})
            MERGE (e)-[:BELONGS_TO]->(d)
            WITH e, d
            MERGE (t:Team {name: $team})
            MERGE (e)-[:MEMBER_OF]->(t)
            MERGE (t)-[:BELONGS_TO]->(d)
            """
            
            # Convert data types and handle missing values
            try:
                salary = float(employee_data.get('Salary', 0))
            except:
                salary = 0
                
            try:
                bonus = float(employee_data.get('Bonus %', 0).strip('%')) / 100
            except:
                bonus = 0
            
            # Process senior management boolean
            is_senior_management = str(employee_data.get('Senior Management', '')).upper() == 'TRUE'
            
            # Process the data
            session.run(cypher_query, 
                       id=str(employee_id),
                       name=str(employee_data.get('First Name', f"Employee_{employee_id}")),
                       gender=str(employee_data.get('Gender', 'Unknown')),
                       start_date=str(employee_data.get('Start Date', '2023-01-01')),
                       last_login=str(employee_data.get('Last Login Time', '')),
                       salary=salary,
                       bonus=bonus,
                       is_senior_management=is_senior_management,
                       team=str(employee_data.get('Team', 'General')),
                       department=str(employee_data.get('Department', 'General')))

    def create_department_relationships(self):
        with self.driver.session() as session:
            # Create relationships between employees in the same department
            cypher_query = """
            MATCH (e1:Employee)-[:BELONGS_TO]->(d:Department)<-[:BELONGS_TO]-(e2:Employee)
            WHERE e1.id < e2.id
            MERGE (e1)-[:WORKS_WITH]->(e2)
            """
            session.run(cypher_query)
            
            # Create relationships between team members
            team_query = """
            MATCH (e1:Employee)-[:MEMBER_OF]->(t:Team)<-[:MEMBER_OF]-(e2:Employee)
            WHERE e1.id < e2.id
            MERGE (e1)-[:TEAMMATES]->(e2)
            """
            session.run(team_query)

def main():
    # Read employee data
    try:
        df = pd.read_csv('employees.csv')
        print(f"Successfully read {len(df)} employee records")
        
        # Add ID column if it doesn't exist
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Initialize Neo4j connection
    try:
        neo4j_connector = Neo4jConnector(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        print("Successfully connected to Neo4j")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return

    try:
        # Clear existing data
        # print("Clearing existing database...")
        # neo4j_connector.clear_database()

        # Create constraints
        print("Creating constraints...")
        neo4j_connector.create_constraints()

        # Process each employee
        print("Processing employees...")
        for index, row in df.iterrows():
            employee_id = row.get('id', index + 1)
            neo4j_connector.create_employee(row, employee_id)
            print(f"Processed employee: {row.get('name', f'Employee_{employee_id}')}")

        # Create department-based relationships
        print("Creating department relationships...")
        neo4j_connector.create_department_relationships()

        print("Data synchronization completed successfully!")

    except Exception as e:
        print(f"Error during data synchronization: {e}")
        raise e  # This will show the full error trace

    finally:
        neo4j_connector.close()

if __name__ == "__main__":
    main()

