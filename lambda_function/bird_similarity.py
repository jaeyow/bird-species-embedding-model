import json

def handler(event, context):
    """
    Lambda function to classify the document using the KNN model
    """
        
    message = "Hello from the bird_similarity (docker image) function"
    print(message)
    print(event)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html'
        },
        'body': json.dumps(event)
    }