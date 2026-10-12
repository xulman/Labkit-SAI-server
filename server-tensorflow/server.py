from fastapi import FastAPI, HTTPException, Body, Response, Depends, Header
from typing import Annotated, Optional
from pathlib import Path
import secrets

import segmentation_methods as NETS
import numpy as np
import tifffile as TIFF
from io import BytesIO
import time

import sys
sys.path.append("../../images_loaders")
import utils


# --- token, read once at startup from secret.txt next to this script ---
SECRET_FILE = Path(__file__).resolve().parent / "secret.txt"
try:
    API_TOKEN = SECRET_FILE.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    sys.exit(f"Missing token file: {SECRET_FILE}")
if not API_TOKEN:
    sys.exit(f"Token file is empty: {SECRET_FILE}")


async def verify_token(x_api_token: Annotated[Optional[str], Header()] = None):
    # constant-time comparison, avoids timing side channel
    if x_api_token is None or not secrets.compare_digest(x_api_token, API_TOKEN):
        # 404 instead of 401/403 => unauthenticated clients learn nothing
        print("refusing some connection!")
        raise HTTPException(status_code=404, detail="Not Found")
##
## Requires curl commands with:
##      -H "X-API-Token: $(cat secret.txt)"
##
## (note the subshell `cat` so that the secret itself never appears in bash history)


methods_folder = '../server-MODELS'
methods = NETS.SegmentationMethods(methods_folder)

app = FastAPI(
    docs_url=None,       # no /docs (Swagger UI)
    redoc_url=None,      # no /redoc
    openapi_url=None,    # no /openapi.json -- the actual discovery source
    dependencies=[Depends(verify_token)],  # applies to every route
)

@app.get("/")
async def handle_root():
    return {"Welcome message": "Hello World. This is a small server of 2D cell segmentation models."}


@app.get("/segmentation_2D/list_available_methods")
async def list_methods():
    return {"status": "OK", "available_network_names": [net for net in sorted(methods.list_avail_methods())]}


@app.get("/segmentation_2D/update_available_methods_list")
async def update_methods():
    methods.rescan_methods(methods_folder)
    return {"status": "OK", "available_network_names": [net for net in sorted(methods.list_avail_methods())]}


class OctetStreamResponse(Response):
    media_type='application/octet-stream'

@app.post("/segmentation_2D/on_posted_tiff/use/{method_name}", response_class=OctetStreamResponse)
async def segment_2d_already_normalized_tiff(tiff_image: Annotated[bytes, Body(media_type='application/octet-stream')], method_name: str):
    '''
    The POST data must be a `serialized TIFF` image.

    Returned stream will be also a `serialized TIFF` image.
    The input and output images are 2D and of the same width
    and height.

    How to test:
    ```
    curl -X POST --data-binary @test.tif -H "Content-Type: application/octet-stream"  http://127.0.0.1:8000/segmentation_2D/on_posted_tiff/use/cellpose --output res_image_stream.tif
    ```
    '''
    return segment_2d_tiff(tiff_image, do_normalization = False, method_name = method_name)


@app.post("/segmentation_2D/on_posted_tiff/normalize_it_then_use/{method_name}", response_class=OctetStreamResponse)
async def segment_2d_plain_tiff(tiff_image: Annotated[bytes, Body(media_type='application/octet-stream')], method_name: str):
    '''
    The POST data must be a `serialized TIFF` image.

    Returned stream will be also a `serialized TIFF` image.
    The input and output images are 2D and of the same width
    and height.

    The submitted image will be normalized to 0-1 range prior the segmentation.

    How to test:
    ```
    curl -X POST --data-binary @test.tif -H "Content-Type: application/octet-stream"  http://127.0.0.1:8000/segmentation_2D/on_posted_tiff/normalize_it_then_use/cellpose --output res_image_stream.tif
    ```
    '''
    return segment_2d_tiff(tiff_image, do_normalization = True, method_name = method_name)


def segment_2d_tiff(tiff_image: Annotated[bytes, Body(media_type='application/octet-stream')], do_normalization: bool, method_name: str):
    time_start = time.time()

    # check if the requested method is known?
    if method_name not in methods.list_avail_methods():
        raise HTTPException(status_code=400, detail=f"METHOD ERROR: >>{method_name}<< is not available")

    try:
        img = TIFF.imread(BytesIO(tiff_image))
        #print(img, img.shape, img.dtype)

        if do_normalization:
            img = utils.basic_inplace_normalization(img)

        seg_method = methods.get_segmentation_fun(method_name)
        if seg_method is None:
            raise HTTPException(status_code=400, detail=f"METHOD ERROR: >>{method_name}<< is not supported in the encryption module")
        ret_img = seg_method(img)
        #print(ret_img, ret_img.shape, ret_img.dtype)

        ret_buf = BytesIO()
        TIFF.imwrite(ret_buf, ret_img)
        ret_buf.seek(0)
    except Exception as e:
        print(f"general Exception: {e}")
        raise HTTPException(status_code=400, detail=f"GENERAL ERROR: {e}")

    time_stop = time.time()
    print(f"Time needed: {(time_stop - time_start):0.2f} seconds")

    return Response(content=ret_buf.read(), media_type='application/octet-stream' )


@app.post("/segmentation_2D/on_posted_stream_of/{width}/{height}/use/{method_name}", response_class=OctetStreamResponse)
async def segment_2d_already_normalized_stream(stream_image: Annotated[bytes, Body(media_type='application/octet-stream')], width: int, height: int, method_name: str):
    '''
    The POST data must be a `stream of floats` of the total
    length of `width`*`height`. In the stream, the width coordinate
    varies faster (row-major order; x-line after x-line).

    Returned will be a `stream of 16-bit (unsigned) integers`,
    representing similarly (row-major) a 16-bit labelled image
    of the same `width` and `height`.

    How to test:
    ```
    curl -X POST --data-binary @six_floats_test.dat -H "Content-Type: application/octet-stream"  http://127.0.0.1:8000/segmentation_2D/on_posted_stream_of/3/2/use/cellpose --output res_image_stream.dat
    ```
    '''
    return segment_2d_stream(stream_image, width, height, do_normalization = False, method_name = method_name)


@app.post("/segmentation_2D/on_posted_stream_of/{width}/{height}/normalize_it_then_use/{method_name}", response_class=OctetStreamResponse)
async def segment_2d_plain_stream(stream_image: Annotated[bytes, Body(media_type='application/octet-stream')], width: int, height: int, method_name: str):
    '''
    The POST data must be a `stream of floats` of the total
    length of `width`*`height`. In the stream, the width coordinate
    varies faster (row-major order; x-line after x-line).

    Returned will be a `stream of 16-bit (unsigned) integers`,
    representing similarly (row-major) a 16-bit labelled image
    of the same `width` and `height`.

    The submitted image will be normalized to 0-1 range prior the segmentation.

    How to test:
    ```
    curl -X POST --data-binary @six_floats_test.dat -H "Content-Type: application/octet-stream"  http://127.0.0.1:8000/segmentation_2D/on_posted_stream_of/3/2/normalize_it_then_use/cellpose --output res_image_stream.dat
    ```
    '''
    return segment_2d_stream(stream_image, width, height, do_normalization = True, method_name = method_name)


def segment_2d_stream(stream_image: Annotated[bytes, Body(media_type='application/octet-stream')], width: int, height: int, do_normalization: bool, method_name: str, prompt = None):
    time_start = time.time()

    # check if the requested method is known?
    if method_name not in methods.list_avail_methods():
        raise HTTPException(status_code=400, detail=f"METHOD ERROR: >>{method_name}<< is not available")

    try:
        print(f"Expected stream size is {width*height} elements.")
        print(f"Input stream size is {len(stream_image)} bytes.")
        img = np.reshape( np.frombuffer(stream_image, dtype='float32'), (height,width) )
        #print(img, img.shape, img.dtype)
        #TIFF.imwrite("incoming.tif",img)

        if do_normalization:
            img = utils.basic_inplace_normalization(img)

        seg_method = methods.get_segmentation_fun(method_name)
        if seg_method is None:
            raise HTTPException(status_code=400, detail=f"METHOD ERROR: >>{method_name}<< is not supported in the encryption module")
        #TIFF.imwrite("seg-in.tif",img)
        ret_img = seg_method(img, prompt)
        #TIFF.imwrite("seg-out.tif",ret_img)
        #print(ret_img, ret_img.shape, ret_img.dtype)
        print(f"Output stream size is {len(ret_img.tobytes())} bytes.")
    except Exception as e:
        print(f"general Exception: {e}")
        raise HTTPException(status_code=400, detail=f"GENERAL ERROR: {e}")

    time_stop = time.time()
    print(f"Time needed: {(time_stop - time_start):0.2f} seconds")

    return Response(content=ret_img.tobytes(), media_type='application/octet-stream' )


@app.post("/segmentation_2D/on_posted_stream_of/{width}/{height}/use/{method_name}/prompt_box/{x_min}/{y_min}/{x_max}/{y_max}", response_class=OctetStreamResponse)
async def segment_2d_plain_stream_with_prompt(stream_image: Annotated[bytes, Body(media_type='application/octet-stream')],
                                              width: int, height: int, method_name: str,
                                              x_min, y_min, x_max, y_max):
    return segment_2d_stream(stream_image, width, height, do_normalization = False, method_name = method_name, prompt = (x_min, y_min, x_max, y_max))


@app.post("/segmentation_2D/on_posted_stream_of/{width}/{height}/normalize_it_then_use/{method_name}/prompt_box/{x_min}/{y_min}/{x_max}/{y_max}", response_class=OctetStreamResponse)
async def segment_2d_already_normalized_stream_with_prompt(stream_image: Annotated[bytes, Body(media_type='application/octet-stream')],
                                                           width: int, height: int, method_name: str,
                                                           x_min, y_min, x_max, y_max):
    return segment_2d_stream(stream_image, width, height, do_normalization = True, method_name = method_name, prompt = (x_min, y_min, x_max, y_max))


@app.post("/connection_test/data_transfer_times", response_class=OctetStreamResponse)
async def test_transfer_time(stream_of_data: Annotated[bytes, Body(media_type='application/octet-stream')]):
    '''
    POST stream of anything and the same exact content will be sent back,
    all of which is only here to measure the overall up-load and down-load time.
    '''
    # touch explicitly every input byte to simulate store-and-(segment-and-)forward paradigm
    [ val for val in stream_of_data ]
    return Response(content=stream_of_data, media_type='application/octet-stream' )

