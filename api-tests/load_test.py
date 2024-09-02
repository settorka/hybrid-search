# locust_stress_test.py
from locust import HttpUser, TaskSet, task, between
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

class StressTestUser(HttpUser):
    tasks = [SearchTaskSet]
    wait_time = between(0.5, 1)  # Shorter wait times to generate high load
    host = "http://localhost:8000"
