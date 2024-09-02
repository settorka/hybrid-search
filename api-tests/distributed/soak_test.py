# Soak testing checks the system's stability and performance over an extended period under moderate load. 
# This test will simulate a constant moderate load for several hours.
from locust import HttpUser, TaskSet, task, constant_pacing
from faker import Faker
import random

fake = Faker()

class SearchTaskSet(TaskSet):
    @task
    def search(self):
        # Generate a random query using faker
        query = fake.text(max_nb_chars=20)
        query_params = {
            "query": query,
            "top_k": random.randint(1, 20),  # Random top_k between 1 and 20
            "from_": random.randint(0, 100)  # Random from_ between 0 and 100
        }
        self.client.post("/search", json=query_params)

class SoakTestUser(HttpUser):
    tasks = [SearchTaskSet]
    wait_time = constant_pacing(2)  # Users wait exactly 2 seconds between requests
    host = "http://localhost:8000"
