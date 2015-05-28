from django.views.generic import TemplateView
from django.shortcuts import render
from predictionio import EngineClient


def add_colors(node):
    if len(node["children"]) > 0:
        node["children"] = map(add_colors, node["children"])
    if node["yes"] > node["no"] and 2 * node["yes"] > 1 - node["no"]:
        c = 255 - int(node["yes"] * 255.0)
        node["color"] = "#{:02X}FF{:02X}".format(c, c)
    elif node["no"] > node["yes"] and 2 * node["no"] > 1 - node["yes"]:
        c = 255 - int(node["no"] * 255.0)
        node["color"] = "#FF{:02X}{:02X}".format(c, c)
    else:
        node["color"] = "white"
    return node


class IndexView(TemplateView):
    template_name = "sentiment/index.html"

    def post(self, request):
        engine = EngineClient("http://localhost:8001")
        result = engine.send_query({'sentence': request.POST.get('sentence', '')})
        return render(request, 'sentiment/prediction.html', {'node': add_colors(result)})