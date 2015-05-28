from django.views.generic import TemplateView
from django.shortcuts import render
from predictionio import EngineClient


def add_colors(node):
    if len(node["children"]) > 0:
        node["children"] = map(add_colors, node["children"])
    if node["sentiment"] >= 0.0:
        gb = 255 - int(node["sentiment"] * 255.0)
        node["color"] = "#FF{:02X}{:02X}".format(gb, gb)
    else:
        rb = 255 - int(node["sentiment"] * -255.0)
        node["color"] = "#FF{:02X}{:02X}".format(rb, rb)
    return node


class IndexView(TemplateView):
    template_name = "sentiment/index.html"

    def post(self, request):
        engine = EngineClient("http://localhost:8001")
        result = engine.send_query({'sentence': request.POST.get('sentence', '')})
        return render(request, 'sentiment/prediction.html', {'node': add_colors(result)})