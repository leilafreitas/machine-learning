from fastapi.testclient import TestClient
import os
import sys
import pathlib
from src.api.main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# a unit test that tests the status code of the root path
def test_root():
    r = client.get("/")
    assert r.status_code == 200

# a unit test that tests the status code and response 
def test_get_inference_unacc():
    person = {
        "buying": 'low',
        "maint": 'low',
        "doors": '3',
        "persons": '2',
        "lug_boot": 'small',
        "safety": 'high'
    }

    r = client.post("/predict", json=person)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "unacc"

# a unit test that tests the status code and response 
def test_get_inference_acc():
    person = {
        "buying": 'high',
        "maint": 'high',
        "doors": '5more',
        "persons": '4',
        "lug_boot": 'med',
        "safety": 'med'
    }

    r = client.post("/predict", json=person)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "acc"