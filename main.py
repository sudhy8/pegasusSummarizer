from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_text as text

app = FastAPI()

# Load a pre-trained TensorFlow model for text generation
model = tf.saved_model.load('path_to_your_saved_model')

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    input_text = tf.constant([request.text])
    
    summary = model.generate(input_text, 
                             max_length=request.max_length, 
                             min_length=request.min_length)
    
    return {"summary": summary[0].numpy().decode('utf-8')}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)