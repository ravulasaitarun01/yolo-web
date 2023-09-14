# Deep learning libraries
from ultralytics import YOLO
from PIL import Image
import io
import base64
import tempfile
import os

# Web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Starlette()

# Load the YOLO model
model = YOLO('best.pt')

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    # Save byte data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(bytes)
        temp_path = temp_file.name

    # Run inference using YOLO on the temporary file
    results = model(temp_path)
    
    # Convert results to an image
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save("img.jpg")
    
    # Encode the image in base64 to serve in HTML
    img_uri = base64.b64encode(open("img.jpg", 'rb').read()).decode('utf-8')
    
    # Remove the temporary file
    os.remove(temp_path)
    
    return HTMLResponse(
        """
        <html>
            <body>
                <p> Detected Objects: </p>
                <!-- You can add more details about the detections here -->
            </body>
        <figure class="figure">
            <img src="data:image/png;base64, %s" class="figure-img">
        </figure>
        </html>
        """ % img_uri)

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <h1> YOLO Object Detection </h1>
        <p> Upload an image to detect objects. </p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <u> Select picture to upload: </u> <br> <p>
            1. <input type="file" name="file"><br><p>
            2. <input type="submit" value="Upload">
        </form>
        """)

@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8008)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
