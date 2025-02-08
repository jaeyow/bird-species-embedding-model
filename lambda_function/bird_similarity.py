import json


def handler(event, context):
    """
    Endpoint to classify bird species using the KNN model
    """
    message = "Hello from the bird_similarity FastAPI service"
    print(f"Event: {event}")
    print(f"Context: {context}")
    
    return {
        'statusCode': 200,
        'message': message,
        'body': json.dumps(event) if event else "No event data"
    }
    