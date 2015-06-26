from django.views.generic import TemplateView
from django.shortcuts import render
from predictionio import EngineClient


def add_colors(node):
    r = dict()

    if "value" in node:
        r["label"] = node["value"]
    else:
        r["children"] = map(add_colors, [node["left"], node["right"]])

    r["text_color"] = "black"
    vec = node["accum"]["get"]
    c = -vec[0] - 0.5 * vec[1] + 0.5 * vec[3] + vec[4]
    c *= 3
    if c >= 0:
        rb = max(0, 255 - int(c * 255.0))
        print(rb)
        r["color"] = "#{:02X}FF{:02X}".format(rb, rb)
    else:
        gb = max(0, 255 - int(-c * 255.0 * 1.5))
        print(gb)
        r["color"] = "#FF{:02X}{:02X}".format(gb, gb)
        if c <= -0.5:
            r["text_color"] = "white"

    return r


class IndexView(TemplateView):
    template_name = "sentiment/index.html"

    def post(self, request):
        engine = EngineClient("http://localhost:8001")
        result = engine.send_query({'sentence': request.POST.get('sentence', '')})
        return render(request, 'sentiment/prediction.html', {'result': result, 'node': add_colors(result["sentence"])})