var canvas, ctx, flag = false,
    prevX = 0,
    currX = 0,
    prevY = 0,
    currY = 0,
    started = false;

function init() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
    ctx.lineWidth = 25;
    w = canvas.width;
    h = canvas.height;

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}

// function draw() {
//   ctx.beginPath();
//   ctx.moveTo(prevX, prevY);
//   ctx.lineTo(currX, currY);
//   ctx.stroke();
//   ctx.closePath();
// }

function erase() {
    ctx.clearRect(0, 0, w, h);
}

function findxy(res, e) {
    if (e.layerX || e.layerX === 0) {
        e._x = e.layerX;
        e._y = e.layerY;
    } else if (e.offsetX || e.offsetX === 0) {
        e._x = e.offsetX;
        e._y = e.offsetY;
    }
    if (res === 'down') {
        ctx.beginPath();
        ctx.moveTo(e._x, e._y);
        started = true;
    }
    if (res === 'up' || res === "out") {
        if (started) {
            ctx.lineTo(e._x, e._y);
            ctx.stroke();
            started = false;
        }
    }
    if (res === 'move') {
        if (started) {
            ctx.lineTo(e._x, e._y);
            ctx.stroke();
        }
    }
}
