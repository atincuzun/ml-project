from sanic import Sanic
from sanic.response import html
import asyncio
import pathlib

slideshow_root_path = pathlib.Path(__file__).parent.joinpath("slideshow")

# you can find more information about sanic online https://sanicframework.org,
# but you should be good to go with this example code
app = Sanic("slideshow_server")

app.static("/static", slideshow_root_path)


@app.route("/")
async def index(request):
    return html(open(slideshow_root_path.joinpath("slideshow.html"), "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")

    # # ======================== add calls to your model here ======================
    # # uncomment for event emitting demo: the following loop will alternate
    # # emitting events and pausing
    #
    # while True:
    #     print("emitting 'right'")
    #     # app.add_signal(event="right")
    #     await ws.send("right")
    #     await asyncio.sleep(2)
    #
    #     print("emitting 'rotate'")
    #     await ws.send("rotate")
    #     await asyncio.sleep(2)
    #
    #     print("emitting 'left'")
    #     await ws.send("left")
    #     await asyncio.sleep(2)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
