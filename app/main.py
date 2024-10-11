from fastapi import FastAPI
import transformer
import time
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

@app.post("/transformer")
async def tt(dimensionalidad: int, 
                      attention_heads: int, 
                      data: Matrix):
    start = time.time()
    
    # Creamos una capa encoder de Transformer con dimensionalidad n y n cabezas de atención
    encoder_layer = transformer.TransformerEncoderLayer(dimensionalidad, attention_heads)

    input_data = data.matrix

    output_data = encoder_layer.forward(input_data)

    end = time.time()

    var1 = 'Time taken in seconds: '
    var2 = end - start

    str = f'{var1}{var2}\n'.format(var1=var1, var2=var2)
    str1= f"Salida del Transformer Encoder: {output_data} "
    
    return str + str1