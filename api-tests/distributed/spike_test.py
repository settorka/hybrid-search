# Spike testing evaluates how the system handles sudden bursts of load. 
# It starts with a normal load and then introduces a sudden spike.
# run in a test or production env (load balanced)
# 
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

class SpikeTestUser(HttpUser):
    tasks = [SearchTaskSet]
    wait_time = between(1, 5)  # Normal wait times
    host = "http://localhost:8000"
